import logging
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from pathlib import Path
from ..utils import MetricsTop, dict_to_str
from .HingeLoss import HingeLoss


logger = logging.getLogger('MMSA')


def _accumulation_group_size(batch_index, total_batches, accumulation_steps):
    group_start = (batch_index // accumulation_steps) * accumulation_steps
    return min(accumulation_steps, total_batches - group_start)


def _should_optimizer_step(batch_index, total_batches, accumulation_steps):
    return (
        (batch_index + 1) % accumulation_steps == 0
        or batch_index + 1 == total_batches
    )


class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n
        return mse

class DLF():
    def __init__(self, args):
        self.args = args
        self.criterion = nn.L1Loss()           
        self.cosine = nn.CosineEmbeddingLoss()
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)
        self.MSE = MSE()
        self.sim_loss = HingeLoss()

    def _masked_mse(self, prediction, target, padding_mask=None):
        if padding_mask is None:
            return self.MSE(prediction, target)
        valid = (~padding_mask).transpose(0, 1).unsqueeze(-1)
        squared_error = (prediction - target).pow(2) * valid
        denominator = valid.sum().clamp(min=1) * prediction.size(-1)
        return squared_error.sum() / denominator

    def _ccdr_losses(self, output):
        zero = output['output_logit'].new_zeros(())
        if 'conflict_energy' not in output:
            return zero, zero, zero, zero, zero, zero

        padding_masks = output.get('projected_padding_masks', [None] * 3)
        reconstruction_loss = sum(
            self._masked_mse(reconstructed, target, padding_mask)
            for reconstructed, target, padding_mask in zip(
                output['ccdr_reconstructed'],
                output['ccdr_targets'],
                padding_masks,
            )
        )

        def covariance_penalty(left, right, padding_mask):
            if padding_mask is None:
                left = left.reshape(-1, left.size(-1))
                right = right.reshape(-1, right.size(-1))
            else:
                valid = ~padding_mask
                left = left.transpose(0, 1)[valid]
                right = right.transpose(0, 1)[valid]
            left = left - left.mean(dim=0, keepdim=True)
            right = right - right.mean(dim=0, keepdim=True)
            covariance = left.transpose(0, 1).matmul(right)
            covariance = covariance / max(left.size(0) - 1, 1)
            return covariance.pow(2).mean()

        decorrelation_loss = zero
        for consensus, support, conflict, padding_mask in zip(
            output['consensus_parts'],
            output['support_parts'],
            output['conflict_parts'],
            padding_masks,
        ):
            decorrelation_loss = decorrelation_loss + (
                covariance_penalty(consensus, support, padding_mask)
                + covariance_penalty(consensus, conflict, padding_mask)
                + covariance_penalty(support, conflict, padding_mask)
            )

        modality_logits = torch.cat([
            output['logits_l_shared'],
            output['logits_a_shared'],
            output['logits_v_shared'],
        ], dim=1).detach()
        prediction_spread = modality_logits.std(dim=1, unbiased=False)
        prediction_signs = torch.sign(modality_logits)
        sign_disagreement = (
            prediction_signs.max(dim=1).values
            - prediction_signs.min(dim=1).values
        ).abs() * 0.5
        weak_conflict = prediction_spread + sign_disagreement

        conflict_energy = output['conflict_energy'].mean(dim=1)
        target_difference = weak_conflict[:, None] - weak_conflict[None, :]
        energy_difference = conflict_energy[:, None] - conflict_energy[None, :]
        valid_pairs = target_difference.abs() > 1e-4
        if valid_pairs.any():
            ranking_loss = torch.relu(
                0.1 - target_difference.sign() * energy_difference
            )[valid_pairs].mean()
        else:
            ranking_loss = zero

        low_conflict_weight = torch.exp(-weak_conflict)
        delta_loss = (
            low_conflict_weight * output['conflict_delta'].view(-1).abs()
        ).mean()

        conflict_ratios = []
        route_values = []
        for gate, padding_mask in zip(output['route_gates'], padding_masks):
            if padding_mask is None:
                conflict_ratios.append((1.0 - gate).mean(dim=(0, 2)))
                route_values.append(gate.reshape(-1))
            else:
                valid = (~padding_mask).transpose(0, 1).unsqueeze(-1)
                denominator = valid.sum(dim=(0, 2)).clamp(min=1) * gate.size(-1)
                conflict_ratios.append(((1.0 - gate) * valid).sum(dim=(0, 2)) / denominator)
                route_values.append(gate.transpose(0, 1)[~padding_mask].reshape(-1))
        conflict_ratio = torch.stack(conflict_ratios, dim=1).mean(dim=1)
        routing_target = weak_conflict / (1.0 + weak_conflict)
        routing_loss = self.MSE(conflict_ratio, routing_target)
        route_values = torch.cat(route_values, dim=0)
        minimum_variance = getattr(self.args, 'cred_route_min_variance', 0.01)
        routing_loss = routing_loss + torch.relu(
            route_values.new_tensor(minimum_variance)
            - route_values.var(unbiased=False)
        )

        consensus_logit = output['consensus_logit'].detach().view(-1)
        consensus_direction = torch.sign(consensus_logit)
        valid_direction = consensus_logit.abs() > 0.1
        if valid_direction.any():
            support_effect_loss = torch.relu(
                -consensus_direction * output['support_delta'].view(-1)
            )[valid_direction].mean()
            conflict_effect_loss = (
                routing_target * torch.relu(
                    consensus_direction * output['conflict_delta'].view(-1)
                )
            )[valid_direction].mean()
            effect_loss = support_effect_loss + conflict_effect_loss
        else:
            effect_loss = zero
        return (
            reconstruction_loss,
            decorrelation_loss,
            ranking_loss,
            delta_loss,
            routing_loss,
            effect_loss,
        )

    def do_train(self, model, dataloader, return_epoch_results=False):

        # 0: DLF model
        params = model[0].parameters()
        if self.args.use_bert and self.args.use_finetune:
            bert_params, other_params = [], []
            for name, parameter in model[0].named_parameters():
                if name.startswith('text_model.model.'):
                    bert_params.append(parameter)
                else:
                    other_params.append(parameter)
            bert_lr = getattr(self.args, 'bert_learning_rate', 2e-5)
            optimizer = optim.AdamW(
                [
                    {'params': bert_params, 'lr': bert_lr},
                    {'params': other_params, 'lr': self.args.learning_rate},
                ],
                weight_decay=self.args.weight_decay,
            )
            logger.info(
                "Optimizer: AdamW, bert_lr=%s, other_lr=%s, weight_decay=%s",
                bert_lr,
                self.args.learning_rate,
                self.args.weight_decay,
            )
        else:
            optimizer = optim.Adam(params, lr=self.args.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=self.args.patience)

        epochs, best_epoch = 0, 0
        if return_epoch_results:
            epoch_results = {
                'train': [],
                'valid': [],
                'test': []
            }
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0

        net = []
        net_DLF = model[0]
        net.append(net_DLF)    
        model = net
        
        while True:
            epochs += 1
            y_pred, y_true = [], []
            for mod in model:
                mod.train()
              

            train_loss = 0.0
            accumulation_steps = self.args.update_epochs
            optimizer.zero_grad()
            with tqdm(dataloader['train']) as td:
                for batch_index, batch_data in enumerate(td):
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    padding_mask = batch_data.get('padding_mask')
                    if padding_mask is not None:
                        padding_mask = padding_mask.to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    labels = labels.view(-1, 1)

                    output = model[0](text, audio, vision, padding_mask)

                    # task loss
                    loss_task_all = self.criterion(output['output_logit'], labels)
                    loss_task_shared = (
                        self.criterion(output['logits_l_shared'], labels)
                        + self.criterion(output['logits_v_shared'], labels)
                        + self.criterion(output['logits_a_shared'], labels)
                    )
                    shared_task_weight = (
                        getattr(self.args, 'ccdr_shared_task_weight', 0.5)
                        if 'conflict_energy' in output else 0.0
                    )
                    consensus_task_loss = (
                        self.criterion(output['consensus_logit'], labels)
                        if 'consensus_logit' in output else loss_task_all.new_zeros(())
                    )
                    
                    # total MSA loss L_msa
                    loss_task = (
                        loss_task_all
                        + shared_task_weight * loss_task_shared
                        + getattr(self.args, 'cred_consensus_task_weight', 0.5) * consensus_task_loss
                    )
                    
                    # reconstruction loss L_r
                    padding_masks = output['projected_padding_masks']
                    loss_recon_l = self._masked_mse(output['recon_l'], output['origin_l'], padding_masks[0])
                    loss_recon_a = self._masked_mse(output['recon_a'], output['origin_a'], padding_masks[1])
                    loss_recon_v = self._masked_mse(output['recon_v'], output['origin_v'], padding_masks[2])
                    loss_recon = loss_recon_l + loss_recon_v + loss_recon_a

                    # specific loss L_s 
                    loss_sl_slr = self._masked_mse(output['s_l'], output['s_l_r'], padding_masks[0])
                    loss_sa_sla = self._masked_mse(output['s_a'], output['s_a_r'], padding_masks[1])
                    loss_sv_slv = self._masked_mse(output['s_v'], output['s_v_r'], padding_masks[2])
                    loss_s_sr = loss_sl_slr + loss_sv_slv + loss_sa_sla

                    # ort loss L_o
                    num = self.args.dst_feature_dim_nheads[0]

                    s_l_flat = output['s_l'].reshape(-1, num)
                    s_v_flat = output['s_v'].reshape(-1, num)
                    s_a_flat = output['s_a'].reshape(-1, num)
                    cosine_similarity_s_c_l = self.cosine(s_l_flat, output['c_l'].reshape(-1, num), -torch.ones(s_l_flat.size(0), device=s_l_flat.device))
                    cosine_similarity_s_c_v = self.cosine(s_v_flat, output['c_v'].reshape(-1, num), -torch.ones(s_v_flat.size(0), device=s_v_flat.device))
                    cosine_similarity_s_c_a = self.cosine(s_a_flat, output['c_a'].reshape(-1, num), -torch.ones(s_a_flat.size(0), device=s_a_flat.device))
                    
                    loss_ort = cosine_similarity_s_c_l + cosine_similarity_s_c_v + cosine_similarity_s_c_a

                    # triplet margin loss L_m
                    c_l, c_v, c_a = output['c_l_sim'], output['c_v_sim'], output['c_a_sim']
                    ids, feats = [], []
                    for i in range(labels.size(0)):
                        feats.append(c_l[i].view(1, -1))
                        feats.append(c_v[i].view(1, -1))
                        feats.append(c_a[i].view(1, -1))
                        ids.append(labels[i].view(1, -1))
                        ids.append(labels[i].view(1, -1))
                        ids.append(labels[i].view(1, -1))
                    feats = torch.cat(feats, dim=0)
                    ids = torch.cat(ids, dim=0)
                    loss_sim = self.sim_loss(ids, feats)

                    (
                        loss_ccdr_recon,
                        loss_ccdr_decor,
                        loss_ccdr_rank,
                        loss_ccdr_delta,
                        loss_cred_routing,
                        loss_cred_effect,
                    ) = self._ccdr_losses(output)
                    loss_ccdr = (
                        getattr(self.args, 'ccdr_reconstruction_weight', 0.1) * loss_ccdr_recon
                        + getattr(self.args, 'ccdr_decorrelation_weight', 0.01) * loss_ccdr_decor
                        + getattr(self.args, 'ccdr_ranking_weight', 0.1) * loss_ccdr_rank
                        + getattr(self.args, 'ccdr_delta_weight', 0.05) * loss_ccdr_delta
                        + getattr(self.args, 'cred_routing_weight', 0.1) * loss_cred_routing
                        + getattr(self.args, 'cred_effect_weight', 0.05) * loss_cred_effect
                    )
                    #overall loss L_DLF
                    combined_loss = loss_task + (loss_s_sr + loss_recon + (loss_sim+loss_ort) * 0.1) * 0.1 + loss_ccdr

                    train_loss += combined_loss.item()
                    y_pred.append(output['output_logit'].cpu())
                    y_true.append(labels.cpu())

                    total_batches = len(dataloader['train'])
                    group_size = _accumulation_group_size(
                        batch_index, total_batches, accumulation_steps
                    )
                    (combined_loss / group_size).backward()
                    is_group_end = _should_optimizer_step(
                        batch_index, total_batches, accumulation_steps
                    )
                    if is_group_end:
                        if self.args.grad_clip != -1.0:
                            nn.utils.clip_grad_value_(model[0].parameters(), self.args.grad_clip)
                        optimizer.step()
                        optimizer.zero_grad()
            

            train_loss = train_loss / len(dataloader['train'])
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)
            logger.info(
                f">> Epoch: {epochs} "
                f"TRAIN -({self.args.model_name}) [{epochs - best_epoch}/{epochs}/{self.args.cur_seed}] "
                f">> total_loss: {round(train_loss, 4)} "
                f"{dict_to_str(train_results)}"
            )
            # validation
            val_results = self.do_test(model[0], dataloader['valid'], mode="VAL")
            test_results = self.do_test(model[0], dataloader['test'], mode="TEST")
            cur_valid = val_results[self.args.KeyEval]
            scheduler.step(val_results['Loss'])
            model_save_path = Path(self.args.model_save_path)
            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                torch.save(model[0].state_dict(), model_save_path)

            if return_epoch_results:
                train_results["Loss"] = train_loss
                epoch_results['train'].append(train_results)
                epoch_results['valid'].append(val_results)
                test_results = self.do_test(model, dataloader['test'], mode="TEST")
                epoch_results['test'].append(test_results)
            # early stop
            if epochs - best_epoch >= self.args.early_stop:
                return epoch_results if return_epoch_results else None
            if epochs >= getattr(self.args, 'max_epochs', float('inf')):
                return epoch_results if return_epoch_results else None

    def do_test(self, model, dataloader, mode="VAL", return_sample_results=False):

        model.eval()
        y_pred, y_true = [], []

        eval_loss = 0.0
        if return_sample_results:
            ids, sample_results = [], []
            all_labels = []
            features = {
                "Feature_t": [],
                "Feature_a": [],
                "Feature_v": [],
                "Feature_f": [],
            }

        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    padding_mask = batch_data.get('padding_mask')
                    if padding_mask is not None:
                        padding_mask = padding_mask.to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    labels = labels.view(-1, 1)
                    output = model(text, audio, vision, padding_mask)
                    loss = self.criterion(output['output_logit'], labels)
                    eval_loss += loss.item()
                    y_pred.append(output['output_logit'].cpu())
                    y_true.append(labels.cpu())

        eval_loss = eval_loss / len(dataloader)
        pred, true = torch.cat(y_pred), torch.cat(y_true)

        eval_results = self.metrics(pred, true)
        eval_results["Loss"] = round(eval_loss, 4)
        logger.info(f"{mode}-({self.args.model_name}) >> {dict_to_str(eval_results)}")

        if return_sample_results:
            eval_results["Ids"] = ids
            eval_results["SResults"] = sample_results
            for k in features.keys():
                features[k] = np.concatenate(features[k], axis=0)
            eval_results['Features'] = features
            eval_results['Labels'] = all_labels

        return eval_results