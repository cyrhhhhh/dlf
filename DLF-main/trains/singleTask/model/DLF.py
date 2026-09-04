"""
here is the mian backbone for DLF
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...subNets import BertTextEncoder
from ...subNets.transformers_encoder.transformer import TransformerEncoder   
from .CCDR import CCDR

class DLF(nn.Module):
    def __init__(self, args):
        super(DLF, self).__init__()
        if args.use_bert:
            self.text_model = BertTextEncoder(use_finetune=args.use_finetune, transformers=args.transformers,
                                              pretrained=args.pretrained)
        self.use_bert = args.use_bert
        dst_feature_dims, nheads = args.dst_feature_dim_nheads
        if 'seq_lens' in args:
            self.len_l, self.len_a, self.len_v = args.seq_lens
        elif args.dataset_name == 'mosi':
            if args.need_data_aligned:
                self.len_l, self.len_v, self.len_a = 50, 50, 50
            else:
                self.len_l, self.len_v, self.len_a = 50, 500, 375
        if args.dataset_name == 'mosei':
            if args.need_data_aligned:
                self.len_l, self.len_v, self.len_a = 50, 50, 50
            else:
                self.len_l, self.len_v, self.len_a = 50, 500, 500
        self.orig_d_l, self.orig_d_a, self.orig_d_v = args.feature_dims
        self.d_l = self.d_a = self.d_v = dst_feature_dims
        self.num_heads = nheads     
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
        self.use_ccdr = getattr(args, 'use_ccdr', True)
        self.use_lfa_aux = getattr(args, 'use_lfa_aux', True)
        self.use_specific_aux = getattr(args, 'use_specific_aux', True)
        self.ccdr = CCDR(
            hidden_dim=self.d_l,
            num_heads=self.num_heads,
            dropout=getattr(args, 'ccdr_dropout', self.output_dropout),
            route_temperature=getattr(args, 'cred_route_temperature', 1.0),
        )
        combined_dim_low = self.d_a    
        combined_dim_high = self.d_a 
        combined_dim = (self.d_l + self.d_a + self.d_v ) + self.d_l * 3  
        
        output_dim = 1

        # 1. Temporal convolutional layers for initial feature
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=args.conv1d_kernel_size_l, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=args.conv1d_kernel_size_a, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=args.conv1d_kernel_size_v, padding=0, bias=False)

        # 2. Modality-specific encoder
        self.encoder_s_l = self.get_network(self_type='l', layers = self.layers)       
        self.encoder_s_v = self.get_network(self_type='v', layers = self.layers)
        self.encoder_s_a = self.get_network(self_type='a', layers = self.layers)

        #   Modality-shared encoder 
        self.encoder_c = self.get_network(self_type='l', layers = self.layers)        
        

        # 3. Decoder for reconstruct three modalities
        self.decoder_l = nn.Conv1d(self.d_l * 2, self.d_l, kernel_size=1, padding=0, bias=False)     
        self.decoder_v = nn.Conv1d(self.d_v * 2, self.d_v, kernel_size=1, padding=0, bias=False)
        self.decoder_a = nn.Conv1d(self.d_a * 2, self.d_a, kernel_size=1, padding=0, bias=False)

        # for align c_l, c_v, c_a
        self.align_c_l = nn.Linear(combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1), combined_dim_low)
        self.align_c_v = nn.Linear(combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1), combined_dim_low)
        self.align_c_a = nn.Linear(combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1), combined_dim_low)

        self.self_attentions_c_l = self.get_network(self_type='l')
        self.self_attentions_c_v = self.get_network(self_type='v')
        self.self_attentions_c_a = self.get_network(self_type='a')

        self.proj1_c = nn.Linear(self.d_l * 3, self.d_l * 3)
        self.proj2_c = nn.Linear(self.d_l * 3, self.d_l * 3)
        self.out_layer_c = nn.Linear(self.d_l * 3, output_dim)


        # 4 Multimodal Crossmodal Attentions
        self.trans_l_with_a = self.get_network(self_type='la', layers = self.layers)  
        self.trans_l_with_v = self.get_network(self_type='lv', layers = self.layers) 
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=self.layers)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)


        # 5. fc layers for shared features
        self.proj1_l_low = nn.Linear(combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1), combined_dim_low)
        self.proj2_l_low = nn.Linear(combined_dim_low, combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1))
        self.out_layer_l_low = nn.Linear(combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1), output_dim)
        self.proj1_v_low = nn.Linear(combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1), combined_dim_low)
        self.proj2_v_low = nn.Linear(combined_dim_low, combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1))
        self.out_layer_v_low = nn.Linear(combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1), output_dim)
        self.proj1_a_low = nn.Linear(combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1), combined_dim_low)
        self.proj2_a_low = nn.Linear(combined_dim_low, combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1))
        self.out_layer_a_low = nn.Linear(combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1), output_dim)

        
        # 6. fc layers for specific features
        self.proj1_l_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.proj2_l_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.out_layer_l_high = nn.Linear(combined_dim_high, output_dim)
        self.proj1_v_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.proj2_v_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.out_layer_v_high = nn.Linear(combined_dim_high, output_dim)
        self.proj1_a_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.proj2_a_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.out_layer_a_high = nn.Linear(combined_dim_high, output_dim)
        
        # 7. project for fusion
        self.projector_l = nn.Linear(self.d_l, self.d_l)
        self.projector_v = nn.Linear(self.d_v, self.d_v)
        self.projector_a = nn.Linear(self.d_a, self.d_a)
        self.projector_c = nn.Linear(3 * self.d_l, 3 * self.d_l)
        
        # 8. final project
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)
        
    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v        
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = self.d_v, self.attn_dropout
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)


    def forward(self, text, audio, video):
        #extraction
        if self.use_bert:
            text = self.text_model(text)
        x_l = F.dropout(text.transpose(1, 2), p=self.text_dropout, training=self.training) 
        x_a = audio.transpose(1, 2)
        x_v = video.transpose(1, 2)
        

        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l) 
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a) 
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        
        proj_x_l = proj_x_l.permute(2, 0, 1)   
        proj_x_v = proj_x_v .permute(2, 0, 1)  
        proj_x_a = proj_x_a.permute(2, 0, 1)

        #disentanglement
        s_l = self.encoder_s_l(proj_x_l)    
        s_v = self.encoder_s_v(proj_x_v)
        s_a = self.encoder_s_a(proj_x_a)

        c_l = self.encoder_c(proj_x_l)
        c_v = self.encoder_c(proj_x_v)
        c_a = self.encoder_c(proj_x_a)


        s_l = s_l.permute(1, 2, 0)   
        s_v = s_v.permute(1, 2, 0)
        s_a = s_a.permute(1, 2, 0)

        c_l = c_l.permute(1, 2, 0)
        c_v = c_v.permute(1, 2, 0)
        c_a = c_a.permute(1, 2, 0)
        c_list = [c_l, c_v, c_a]


        c_l_sim = self.align_c_l(c_l.contiguous().view(x_l.size(0), -1))
        c_v_sim = self.align_c_v(c_v.contiguous().view(x_l.size(0), -1))
        c_a_sim = self.align_c_a(c_a.contiguous().view(x_l.size(0), -1))
        
        recon_l = self.decoder_l(torch.cat([s_l, c_list[0]], dim=1))
        recon_v = self.decoder_v(torch.cat([s_v, c_list[1]], dim=1))
        recon_a = self.decoder_a(torch.cat([s_a, c_list[2]], dim=1))

        recon_l = recon_l.permute(2, 0, 1)  
        recon_v = recon_v.permute(2, 0, 1)   
        recon_a = recon_a.permute(2, 0, 1)

        s_l_r = self.encoder_s_l(recon_l).permute(1, 2, 0)                                                                                             
        s_v_r = self.encoder_s_v(recon_v).permute(1, 2, 0)
        s_a_r = self.encoder_s_a(recon_a).permute(1, 2, 0)
        
        s_l = s_l.permute(2, 0, 1)  
        s_v = s_v.permute(2, 0, 1)   
        s_a = s_a.permute(2, 0, 1)

        c_l = c_l.permute(2, 0, 1)
        c_v = c_v.permute(2, 0, 1)
        c_a = c_a.permute(2, 0, 1)

        ccdr_output = None
        if self.use_ccdr:
            ccdr_output = self.ccdr(
                shared_features=[c_l, c_a, c_v],
                specific_features=[s_l, s_a, s_v],
            )
       
       #enhancement
        hs_l_low = c_l.transpose(0, 1).contiguous().view(x_l.size(0), -1)  
        repr_l_low = self.proj1_l_low(hs_l_low)                            
        hs_proj_l_low = self.proj2_l_low(
            F.dropout(F.relu(repr_l_low, inplace=True), p=self.output_dropout, training=self.training))
        hs_proj_l_low += hs_l_low         
        logits_l_low = self.out_layer_l_low(hs_proj_l_low)

        hs_v_low = c_v.transpose(0, 1).contiguous().view(x_v.size(0), -1)
        repr_v_low = self.proj1_v_low(hs_v_low)
        hs_proj_v_low = self.proj2_v_low(
            F.dropout(F.relu(repr_v_low, inplace=True), p=self.output_dropout, training=self.training))
        hs_proj_v_low += hs_v_low
        logits_v_low = self.out_layer_v_low(hs_proj_v_low)

        hs_a_low = c_a.transpose(0, 1).contiguous().view(x_a.size(0), -1)
        repr_a_low = self.proj1_a_low(hs_a_low)
        hs_proj_a_low = self.proj2_a_low(
            F.dropout(F.relu(repr_a_low, inplace=True), p=self.output_dropout, training=self.training))
        hs_proj_a_low += hs_a_low
        logits_a_low = self.out_layer_a_low(hs_proj_a_low)

        
        c_l_att = self.self_attentions_c_l(c_l)  
        if type(c_l_att) == tuple:
            c_l_att = c_l_att[0]
        c_l_att = c_l_att[-1]

        c_v_att = self.self_attentions_c_v(c_v)
        if type(c_v_att) == tuple:
            c_v_att = c_v_att[0]
        c_v_att = c_v_att[-1]

        c_a_att = self.self_attentions_c_a(c_a)
        if type(c_a_att) == tuple:
            c_a_att = c_a_att[0]
        c_a_att = c_a_att[-1]

        c_fusion = torch.cat([c_l_att, c_v_att, c_a_att], dim=1)   

        c_proj = self.proj2_c(
            F.dropout(F.relu(self.proj1_c(c_fusion), inplace=True), p=self.output_dropout,
                      training=self.training))
        c_proj += c_fusion                       
        logits_c = self.out_layer_c(c_proj)     
        
        run_lfa = not self.use_ccdr or self.use_lfa_aux
        hetero_output = None
        if run_lfa:
            h_ls = self.trans_l_mem(s_l)
            if type(h_ls) == tuple:
                h_ls = h_ls[0]
            last_h_l = h_ls[-1]

            h_as = self.trans_a_mem(self.trans_l_with_a(s_l, s_a, s_a))
            if type(h_as) == tuple:
                h_as = h_as[0]
            last_h_a = h_as[-1]

            h_vs = self.trans_v_mem(self.trans_l_with_v(s_l, s_v, s_v))
            if type(h_vs) == tuple:
                h_vs = h_vs[0]
            last_h_v = h_vs[-1]

            hs_proj_l_high = self.proj2_l_high(
                F.dropout(F.relu(self.proj1_l_high(last_h_l), inplace=True), p=self.output_dropout, training=self.training))
            hs_proj_l_high += last_h_l
            hs_proj_v_high = self.proj2_v_high(
                F.dropout(F.relu(self.proj1_v_high(last_h_v), inplace=True), p=self.output_dropout, training=self.training))
            hs_proj_v_high += last_h_v
            hs_proj_a_high = self.proj2_a_high(
                F.dropout(F.relu(self.proj1_a_high(last_h_a), inplace=True), p=self.output_dropout,
                          training=self.training))
            hs_proj_a_high += last_h_a
            hetero_output = {
                'features': [hs_proj_l_high, hs_proj_a_high, hs_proj_v_high],
                'logits_l_hetero': self.out_layer_l_high(hs_proj_l_high),
                'logits_a_hetero': self.out_layer_a_high(hs_proj_a_high),
                'logits_v_hetero': self.out_layer_v_high(hs_proj_v_high),
            }
        elif self.use_ccdr and self.use_specific_aux and self.training:
            specific_l = s_l.mean(dim=0)
            specific_a = s_a.mean(dim=0)
            specific_v = s_v.mean(dim=0)
            hs_proj_l_high = self.proj2_l_high(
                F.dropout(F.relu(self.proj1_l_high(specific_l), inplace=True), p=self.output_dropout, training=True))
            hs_proj_l_high += specific_l
            hs_proj_a_high = self.proj2_a_high(
                F.dropout(F.relu(self.proj1_a_high(specific_a), inplace=True), p=self.output_dropout, training=True))
            hs_proj_a_high += specific_a
            hs_proj_v_high = self.proj2_v_high(
                F.dropout(F.relu(self.proj1_v_high(specific_v), inplace=True), p=self.output_dropout, training=True))
            hs_proj_v_high += specific_v
            hetero_output = {
                'logits_l_hetero': self.out_layer_l_high(hs_proj_l_high),
                'logits_a_hetero': self.out_layer_a_high(hs_proj_a_high),
                'logits_v_hetero': self.out_layer_v_high(hs_proj_v_high),
            }
        
        anchor_output = logits_c
        if hetero_output is not None and 'features' in hetero_output:
            hs_proj_l_high, hs_proj_a_high, hs_proj_v_high = hetero_output['features']
            last_h_l = torch.sigmoid(self.projector_l(hs_proj_l_high))
            last_h_v = torch.sigmoid(self.projector_v(hs_proj_v_high))
            last_h_a = torch.sigmoid(self.projector_a(hs_proj_a_high))
            c_fusion = torch.sigmoid(self.projector_c(c_fusion))
            last_hs = torch.cat([last_h_l, last_h_v, last_h_a, c_fusion], dim=1)
            last_hs_proj = self.proj2(
                F.dropout(F.relu(self.proj1(last_hs), inplace=True), p=self.output_dropout, training=self.training))
            last_hs_proj += last_hs
            anchor_output = self.out_layer(last_hs_proj)

        if ccdr_output is not None:
            modality_logits = torch.cat([
                logits_l_low,
                logits_a_low,
                logits_v_low,
            ], dim=1)
            prediction_spread = modality_logits.std(dim=1, unbiased=False)
            prediction_signs = torch.sign(modality_logits)
            sign_disagreement = (
                prediction_signs.max(dim=1).values
                - prediction_signs.min(dim=1).values
            ).abs() * 0.5
            conflict_gate = (
                prediction_spread / (1.0 + prediction_spread)
                + 0.5 * sign_disagreement
            ).clamp(max=1.0).detach().unsqueeze(-1)
            bounded_correction = torch.tanh(ccdr_output['conflict_delta'])
            output = anchor_output + conflict_gate * bounded_correction
        else:
            output = anchor_output

        res = {
            'origin_l': proj_x_l,
            'origin_v': proj_x_v,
            'origin_a': proj_x_a,
            's_l': s_l,        
            's_v': s_v,
            's_a': s_a,
            'c_l': c_l,
            'c_v': c_v,
            'c_a': c_a,
            's_l_r': s_l_r,
            's_v_r': s_v_r,
            's_a_r': s_a_r,
            'recon_l': recon_l,
            'recon_v': recon_v,
            'recon_a': recon_a,
            'c_l_sim': c_l_sim,
            'c_v_sim': c_v_sim,
            'c_a_sim': c_a_sim,
            'logits_l_shared': logits_l_low,
            'logits_v_shared': logits_v_low,
            'logits_a_shared': logits_a_low,
            'logits_c': logits_c,
            'output_logit': output
        }
        if hetero_output is not None:
            res.update({
                'logits_l_hetero': hetero_output['logits_l_hetero'],
                'logits_v_hetero': hetero_output['logits_v_hetero'],
                'logits_a_hetero': hetero_output['logits_a_hetero'],
            })
        if ccdr_output is not None:
            res.update({
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
                'anchor_logit': anchor_output,
                'conflict_gate': conflict_gate,
                'support_scale': ccdr_output['support_scale'],
                'conflict_scale': ccdr_output['conflict_scale'],
                'branch_attentions': ccdr_output['branch_attentions'],
                'ccdr_reconstructed': ccdr_output['reconstructed_parts'],
                'ccdr_targets': ccdr_output['evidence_targets'],
                'evidence_attention': ccdr_output['evidence_attention'],
            })
        return res
