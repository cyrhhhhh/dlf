import torch
import torch.nn as nn
import torch.nn.functional as F

from ...subNets import BertTextEncoder
from ...subNets.transformers_encoder.transformer import TransformerEncoder
from .CCDR import CCDR


class SharedModalityProbe(nn.Module):
    def __init__(self, sequence_length, hidden_dim, dropout):
        super().__init__()
        flattened_dim = sequence_length * hidden_dim
        self.proj_in = nn.Linear(flattened_dim, hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, flattened_dim)
        self.output = nn.Linear(flattened_dim, 1)
        self.dropout = dropout

    def forward(self, features, padding_mask=None):
        features = features.transpose(0, 1)
        if padding_mask is not None:
            features = features.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        flattened = features.reshape(features.size(0), -1)
        hidden = F.relu(self.proj_in(flattened), inplace=True)
        residual = self.proj_out(
            F.dropout(hidden, p=self.dropout, training=self.training)
        )
        return self.output(flattened + residual)


class DLF(nn.Module):
    """Serial shared/private disentanglement followed by CCDR prediction."""

    def __init__(self, args):
        super().__init__()
        if not getattr(args, 'use_ccdr', True):
            raise ValueError('SerialDLF requires use_ccdr=true.')

        self.use_bert = args.use_bert
        if self.use_bert:
            self.text_model = BertTextEncoder(
                use_finetune=args.use_finetune,
                transformers=args.transformers,
                pretrained=args.pretrained,
            )

        self.len_l, self.len_a, self.len_v = args.seq_lens
        self.orig_d_l, self.orig_d_a, self.orig_d_v = args.feature_dims
        hidden_dim, self.num_heads = args.dst_feature_dim_nheads
        self.d_l = self.d_a = self.d_v = hidden_dim
        self.layers = args.nlevels
        self.attn_dropout = args.attn_dropout
        self.attn_dropout_a = args.attn_dropout_a
        self.attn_dropout_v = args.attn_dropout_v
        self.relu_dropout = args.relu_dropout
        self.embed_dropout = args.embed_dropout
        self.res_dropout = args.res_dropout
        self.output_dropout = args.output_dropout
        self.text_dropout = args.text_dropout
        self.attn_mask = args.attn_mask

        self.kernel_l = args.conv1d_kernel_size_l
        self.kernel_a = args.conv1d_kernel_size_a
        self.kernel_v = args.conv1d_kernel_size_v
        self.output_len_l = self.len_l - self.kernel_l + 1
        self.output_len_a = self.len_a - self.kernel_a + 1
        self.output_len_v = self.len_v - self.kernel_v + 1
        if min(self.output_len_l, self.output_len_a, self.output_len_v) <= 0:
            raise ValueError('Conv1d kernel size must not exceed sequence length.')

        self.proj_l = nn.Conv1d(
            self.orig_d_l, hidden_dim, kernel_size=self.kernel_l, bias=False
        )
        self.proj_a = nn.Conv1d(
            self.orig_d_a, hidden_dim, kernel_size=self.kernel_a, bias=False
        )
        self.proj_v = nn.Conv1d(
            self.orig_d_v, hidden_dim, kernel_size=self.kernel_v, bias=False
        )

        self.encoder_s_l = self.get_network('l')
        self.encoder_s_a = self.get_network('a')
        self.encoder_s_v = self.get_network('v')
        self.encoder_c = self.get_network('l')

        self.decoder_l = nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=1, bias=False)
        self.decoder_a = nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=1, bias=False)
        self.decoder_v = nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=1, bias=False)

        self.align_c_l = nn.Linear(self.output_len_l * hidden_dim, hidden_dim)
        self.align_c_a = nn.Linear(self.output_len_a * hidden_dim, hidden_dim)
        self.align_c_v = nn.Linear(self.output_len_v * hidden_dim, hidden_dim)

        self.shared_probe_l = SharedModalityProbe(
            self.output_len_l, hidden_dim, self.output_dropout
        )
        self.shared_probe_a = SharedModalityProbe(
            self.output_len_a, hidden_dim, self.output_dropout
        )
        self.shared_probe_v = SharedModalityProbe(
            self.output_len_v, hidden_dim, self.output_dropout
        )

        self.ccdr = CCDR(
            hidden_dim=hidden_dim,
            num_heads=self.num_heads,
            dropout=getattr(args, 'ccdr_dropout', self.output_dropout),
            route_temperature=getattr(args, 'cred_route_temperature', 1.0),
        )

    def get_network(self, modality):
        dropout = {
            'l': self.attn_dropout,
            'a': self.attn_dropout_a,
            'v': self.attn_dropout_v,
        }[modality]
        return TransformerEncoder(
            embed_dim=self.d_l,
            num_heads=self.num_heads,
            layers=self.layers,
            attn_dropout=dropout,
            relu_dropout=self.relu_dropout,
            res_dropout=self.res_dropout,
            embed_dropout=self.embed_dropout,
            attn_mask=self.attn_mask,
        )

    @staticmethod
    def _project_mask(padding_mask, kernel_size):
        valid = (~padding_mask).float().unsqueeze(1)
        valid_windows = F.max_pool1d(valid, kernel_size=kernel_size, stride=1)
        return valid_windows.squeeze(1) == 0

    @staticmethod
    def _flatten_for_alignment(features, padding_mask):
        features = features.transpose(0, 1)
        features = features.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return features.reshape(features.size(0), -1)

    def forward(self, text, audio, video, padding_mask=None):
        if padding_mask is None:
            if self.use_bert:
                padding_mask = text[:, 1, :].eq(0)
            else:
                padding_mask = text.abs().sum(dim=-1).eq(0)
        padding_mask = padding_mask.bool()

        if self.use_bert:
            text = self.text_model(text)

        x_l = text.transpose(1, 2).masked_fill(padding_mask.unsqueeze(1), 0.0)
        x_l = F.dropout(x_l, p=self.text_dropout, training=self.training)
        x_a = audio.transpose(1, 2).masked_fill(padding_mask.unsqueeze(1), 0.0)
        x_v = video.transpose(1, 2).masked_fill(padding_mask.unsqueeze(1), 0.0)

        mask_l = self._project_mask(padding_mask, self.kernel_l)
        mask_a = self._project_mask(padding_mask, self.kernel_a)
        mask_v = self._project_mask(padding_mask, self.kernel_v)
        padding_masks = [mask_l, mask_a, mask_v]

        proj_x_l = self.proj_l(x_l).masked_fill(mask_l.unsqueeze(1), 0.0)
        proj_x_a = self.proj_a(x_a).masked_fill(mask_a.unsqueeze(1), 0.0)
        proj_x_v = self.proj_v(x_v).masked_fill(mask_v.unsqueeze(1), 0.0)
        proj_x_l = proj_x_l.permute(2, 0, 1)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)

        s_l = self.encoder_s_l(proj_x_l, padding_mask=mask_l)
        s_a = self.encoder_s_a(proj_x_a, padding_mask=mask_a)
        s_v = self.encoder_s_v(proj_x_v, padding_mask=mask_v)
        c_l = self.encoder_c(proj_x_l, padding_mask=mask_l)
        c_a = self.encoder_c(proj_x_a, padding_mask=mask_a)
        c_v = self.encoder_c(proj_x_v, padding_mask=mask_v)

        recon_l = self.decoder_l(torch.cat([s_l, c_l], dim=-1).permute(1, 2, 0))
        recon_a = self.decoder_a(torch.cat([s_a, c_a], dim=-1).permute(1, 2, 0))
        recon_v = self.decoder_v(torch.cat([s_v, c_v], dim=-1).permute(1, 2, 0))
        recon_l = recon_l.permute(2, 0, 1).masked_fill(mask_l.T.unsqueeze(-1), 0.0)
        recon_a = recon_a.permute(2, 0, 1).masked_fill(mask_a.T.unsqueeze(-1), 0.0)
        recon_v = recon_v.permute(2, 0, 1).masked_fill(mask_v.T.unsqueeze(-1), 0.0)

        s_l_r = self.encoder_s_l(recon_l, padding_mask=mask_l)
        s_a_r = self.encoder_s_a(recon_a, padding_mask=mask_a)
        s_v_r = self.encoder_s_v(recon_v, padding_mask=mask_v)

        c_l_sim = self.align_c_l(self._flatten_for_alignment(c_l, mask_l))
        c_a_sim = self.align_c_a(self._flatten_for_alignment(c_a, mask_a))
        c_v_sim = self.align_c_v(self._flatten_for_alignment(c_v, mask_v))
        logits_l_shared = self.shared_probe_l(c_l, mask_l)
        logits_a_shared = self.shared_probe_a(c_a, mask_a)
        logits_v_shared = self.shared_probe_v(c_v, mask_v)

        ccdr_output = self.ccdr(
            shared_features=[c_l, c_a, c_v],
            specific_features=[s_l, s_a, s_v],
            padding_masks=padding_masks,
        )

        result = {
            'origin_l': proj_x_l,
            'origin_a': proj_x_a,
            'origin_v': proj_x_v,
            's_l': s_l,
            's_a': s_a,
            's_v': s_v,
            'c_l': c_l,
            'c_a': c_a,
            'c_v': c_v,
            's_l_r': s_l_r,
            's_a_r': s_a_r,
            's_v_r': s_v_r,
            'recon_l': recon_l,
            'recon_a': recon_a,
            'recon_v': recon_v,
            'c_l_sim': c_l_sim,
            'c_a_sim': c_a_sim,
            'c_v_sim': c_v_sim,
            'logits_l_shared': logits_l_shared,
            'logits_a_shared': logits_a_shared,
            'logits_v_shared': logits_v_shared,
            'projected_padding_masks': padding_masks,
            'output_logit': ccdr_output['output_logit'],
            'consensus_logit': ccdr_output['consensus_logit'],
            'support_delta': ccdr_output['support_delta'],
            'conflict_delta': ccdr_output['conflict_delta'],
            'conflict_energy': ccdr_output['conflict_energy'],
            'consensus_parts': ccdr_output['consensus_parts'],
            'support_parts': ccdr_output['support_parts'],
            'conflict_parts': ccdr_output['conflict_parts'],
            'route_gates': ccdr_output['route_gates'],
            'donor_weights': ccdr_output['donor_weights'],
            'agreement_gates': ccdr_output['agreement_gates'],
            'support_scale': ccdr_output['support_scale'],
            'conflict_scale': ccdr_output['conflict_scale'],
            'branch_attentions': ccdr_output['branch_attentions'],
            'ccdr_reconstructed': ccdr_output['reconstructed_parts'],
            'ccdr_targets': ccdr_output['evidence_targets'],
            'evidence_attention': ccdr_output['evidence_attention'],
        }
        return result