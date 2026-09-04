import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityBalancedAttentionPool(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features, padding_masks=None):
        summaries = []
        temporal_weights = []
        for index, feature in enumerate(features):
            scores = self.scorer(feature)
            if padding_masks is not None:
                scores = scores.masked_fill(
                    padding_masks[index].transpose(0, 1).unsqueeze(-1),
                    float('-inf'),
                )
            weights = torch.softmax(scores, dim=0)
            summaries.append((weights * feature).sum(dim=0))
            temporal_weights.append(weights)
        return torch.stack(summaries, dim=0).mean(dim=0), temporal_weights


class CREDDisentangler(nn.Module):
    """Consensus-residual effect-guided disentanglement."""

    def __init__(self, hidden_dim, num_heads, dropout=0.1, route_temperature=1.0):
        super().__init__()
        self.route_temperature = route_temperature
        self.shared_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(3)
        ])
        self.specific_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(3)
        ])
        self.donor_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=False,
        )
        self.consensus_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(3)
        ])
        self.residual_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(3)
        ])
        self.residual_routers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(3)
        ])
    def forward(self, shared_features, specific_features, padding_masks=None):
        shared_features = [
            norm(feature)
            for norm, feature in zip(self.shared_norms, shared_features)
        ]
        specific_features = [
            norm(feature)
            for norm, feature in zip(self.specific_norms, specific_features)
        ]

        consensus_parts = []
        support_parts = []
        conflict_parts = []
        route_gates = []
        donor_weights = []
        agreement_gates = []
        reconstructed_parts = []

        for index in range(3):
            donor_indices = [other for other in range(3) if other != index]
            query = shared_features[index]
            donor_consensus = []
            donor_scores = []
            for donor_index in donor_indices:
                donor = shared_features[donor_index]
                donor_part, _ = self.donor_attention(
                    query,
                    donor,
                    donor,
                    key_padding_mask=(
                        padding_masks[donor_index]
                        if padding_masks is not None else None
                    ),
                    need_weights=False,
                )
                donor_consensus.append(donor_part)
                donor_scores.append(F.cosine_similarity(
                    query,
                    donor_part,
                    dim=-1,
                ))
            donor_weight = torch.softmax(
                torch.stack(donor_scores, dim=-1) / self.route_temperature,
                dim=-1,
            )
            consensus = (
                donor_weight[..., 0, None] * donor_consensus[0]
                + donor_weight[..., 1, None] * donor_consensus[1]
            )
            consensus = self.consensus_norms[index](query + consensus)

            shared_deviation = shared_features[index] - consensus
            residual = self.residual_projections[index](torch.cat([
                specific_features[index],
                shared_deviation,
            ], dim=-1))
            route_logits = self.residual_routers[index](torch.cat([
                residual,
                consensus,
            ], dim=-1))
            learned_gate = torch.sigmoid(route_logits / self.route_temperature)
            agreement_score = (
                donor_weight * torch.stack(donor_scores, dim=-1)
            ).sum(dim=-1, keepdim=True)
            agreement_gate = (agreement_score.clamp(-1.0, 1.0) + 1.0) * 0.5
            route_gate = 0.5 * (learned_gate + agreement_gate)
            support = route_gate * residual
            conflict = (1.0 - route_gate) * residual
            if padding_masks is not None:
                target_padding = padding_masks[index].transpose(0, 1).unsqueeze(-1)
                consensus = consensus.masked_fill(target_padding, 0.0)
                support = support.masked_fill(target_padding, 0.0)
                conflict = conflict.masked_fill(target_padding, 0.0)
                route_gate = route_gate.masked_fill(target_padding, 0.0)
                agreement_gate = agreement_gate.masked_fill(target_padding, 0.0)

            consensus_parts.append(consensus)
            support_parts.append(support)
            conflict_parts.append(conflict)
            route_gates.append(route_gate)
            donor_weights.append(donor_weight.mean(dim=0))
            agreement_gates.append(agreement_gate)
            reconstructed_parts.append(consensus + support + conflict)

        evidence_targets = [
            shared + specific
            for shared, specific in zip(shared_features, specific_features)
        ]
        return {
            'consensus_parts': consensus_parts,
            'support_parts': support_parts,
            'conflict_parts': conflict_parts,
            'route_gates': route_gates,
            'donor_weights': donor_weights,
            'agreement_gates': agreement_gates,
            'reconstructed_parts': reconstructed_parts,
            'evidence_targets': evidence_targets,
        }


class CCDR(nn.Module):
    """CRED decomposition followed by language-anchored conflict adjudication."""

    def __init__(self, hidden_dim, num_heads, dropout=0.1, route_temperature=1.0):
        super().__init__()
        self.disentangler = CREDDisentangler(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            route_temperature=route_temperature,
        )
        self.consensus_pool = ModalityBalancedAttentionPool(hidden_dim, dropout)
        self.support_pool = ModalityBalancedAttentionPool(hidden_dim, dropout)
        self.conflict_pool = ModalityBalancedAttentionPool(hidden_dim, dropout)
        self.language_pool = ModalityBalancedAttentionPool(hidden_dim, dropout)
        self.support_slot = nn.Parameter(torch.empty(1, 1, hidden_dim))
        self.reverse_slot = nn.Parameter(torch.empty(1, 1, hidden_dim))
        nn.init.normal_(self.support_slot, std=0.02)
        nn.init.normal_(self.reverse_slot, std=0.02)

        self.evidence_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=False,
        )
        self.evidence_norm = nn.LayerNorm(hidden_dim)
        self.consensus_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.support_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.adjudication_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 1),
        )
        self.support_scale = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.conflict_scale = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.support_scale[-2].weight)
        nn.init.zeros_(self.support_scale[-2].bias)
        nn.init.zeros_(self.conflict_scale[-2].weight)
        nn.init.zeros_(self.conflict_scale[-2].bias)

    def forward(self, shared_features, specific_features, padding_masks=None):
        decomposition = self.disentangler(
            shared_features,
            specific_features,
            padding_masks=padding_masks,
        )
        consensus_parts = decomposition['consensus_parts']
        support_parts = decomposition['support_parts']
        conflict_parts = decomposition['conflict_parts']

        consensus_summary, consensus_attention = self.consensus_pool(
            consensus_parts,
            padding_masks,
        )
        support_summary, support_attention = self.support_pool(
            support_parts,
            padding_masks,
        )
        conflict_summary, conflict_attention = self.conflict_pool(
            conflict_parts,
            padding_masks,
        )
        language_consensus, language_attention = self.language_pool([
            consensus_parts[0],
        ], [padding_masks[0]] if padding_masks is not None else None)
        language_anchor, _ = self.language_pool([
            consensus_parts[0] + support_parts[0],
        ], [padding_masks[0]] if padding_masks is not None else None)

        batch_size = shared_features[0].size(1)
        evidence_query = torch.cat([
            self.support_slot.expand(-1, batch_size, -1),
            self.reverse_slot.expand(-1, batch_size, -1),
        ], dim=0)
        evidence_query = evidence_query + (
            language_anchor + consensus_summary
        ).unsqueeze(0)
        nonverbal_conflict = torch.cat(conflict_parts[1:], dim=0)
        evidence, evidence_attention = self.evidence_attention(
            evidence_query,
            nonverbal_conflict,
            nonverbal_conflict,
            key_padding_mask=(
                torch.cat(padding_masks[1:], dim=1)
                if padding_masks is not None else None
            ),
            need_weights=True,
            average_attn_weights=False,
        )
        evidence = self.evidence_norm(evidence + evidence_query)
        support_evidence, reverse_evidence = evidence[0], evidence[1]

        consensus_logit = self.consensus_head(torch.cat([
            consensus_summary,
            language_consensus,
        ], dim=-1))
        support_context = torch.cat([
            consensus_summary,
            support_summary,
        ], dim=-1)
        support_delta_raw = self.support_head(support_context)
        conflict_context = torch.cat([
            consensus_summary,
            language_anchor,
            support_evidence,
            reverse_evidence,
        ], dim=-1)
        conflict_delta_raw = self.adjudication_head(conflict_context)
        support_scale = self.support_scale(support_context)
        conflict_scale = self.conflict_scale(torch.cat([
            consensus_summary,
            conflict_summary,
        ], dim=-1))
        support_delta = support_scale * support_delta_raw
        conflict_delta = conflict_scale * conflict_delta_raw
        output_logit = consensus_logit + support_delta + conflict_delta
        if padding_masks is None:
            conflict_energy = torch.stack([
                conflict.pow(2).mean(dim=(0, 2))
                for conflict in conflict_parts
            ], dim=-1)
        else:
            energies = []
            for conflict, padding_mask in zip(conflict_parts, padding_masks):
                valid = (~padding_mask).transpose(0, 1).unsqueeze(-1)
                denominator = valid.sum(dim=0).squeeze(-1).clamp(min=1)
                energies.append(
                    (conflict.pow(2) * valid).sum(dim=(0, 2))
                    / (denominator * conflict.size(-1))
                )
            conflict_energy = torch.stack(energies, dim=-1)
        return {
            'output_logit': output_logit,
            'consensus_logit': consensus_logit,
            'support_delta': support_delta,
            'conflict_delta': conflict_delta,
            'conflict_energy': conflict_energy,
            'consensus_parts': consensus_parts,
            'support_parts': support_parts,
            'conflict_parts': conflict_parts,
            'route_gates': decomposition['route_gates'],
            'donor_weights': decomposition['donor_weights'],
            'agreement_gates': decomposition['agreement_gates'],
            'support_scale': support_scale,
            'conflict_scale': conflict_scale,
            'branch_attentions': {
                'consensus': consensus_attention,
                'support': support_attention,
                'conflict': conflict_attention,
                'language': language_attention,
            },
            'reconstructed_parts': decomposition['reconstructed_parts'],
            'evidence_targets': decomposition['evidence_targets'],
            'evidence_attention': evidence_attention,
        }