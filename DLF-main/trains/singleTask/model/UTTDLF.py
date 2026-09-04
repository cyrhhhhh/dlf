import torch
import torch.nn as nn

from .CCDR import CCDR


class ResidualMLP(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, features):
        return self.output_norm(features + self.network(features))


class UTTDLF(nn.Module):
    """DLF variant for one pre-extracted vector per modality and utterance."""

    def __init__(self, args):
        super().__init__()
        hidden_dim, num_heads = args.dst_feature_dim_nheads
        dropout = getattr(args, "utterance_dropout", args.output_dropout)
        self.hidden_dim = hidden_dim
        self.use_ccdr = getattr(args, "use_ccdr", True)

        self.input_projections = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_dim),
            )
            for input_dim in args.feature_dims
        ])
        self.shared_encoder = ResidualMLP(hidden_dim, dropout)
        self.specific_encoders = nn.ModuleList([
            ResidualMLP(hidden_dim, dropout) for _ in range(3)
        ])
        self.reconstructors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
            )
            for _ in range(3)
        ])

        self.fusion_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=False,
        )
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.shared_heads = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))
            for _ in range(3)
        ])
        self.specific_heads = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))
            for _ in range(3)
        ])
        self.consensus_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.fusion_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 9),
            nn.Linear(hidden_dim * 9, hidden_dim * 3),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 3, 1),
        )
        self.ccdr = CCDR(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=getattr(args, "ccdr_dropout", dropout),
            route_temperature=getattr(args, "cred_route_temperature", 1.0),
        )

    @staticmethod
    def _flatten_utterance(features, name):
        if features.ndim == 3 and features.size(1) == 1:
            return features[:, 0]
        if features.ndim == 2:
            return features
        raise ValueError(
            f"{name} must have shape (batch, 1, dim) or (batch, dim), "
            f"got {tuple(features.shape)}"
        )

    @staticmethod
    def _as_sequence(features):
        return features.unsqueeze(0)

    def forward(self, text, audio, video):
        inputs = [
            self._flatten_utterance(text, "text"),
            self._flatten_utterance(audio, "audio"),
            self._flatten_utterance(video, "vision"),
        ]
        projected = [
            projection(features)
            for projection, features in zip(self.input_projections, inputs)
        ]
        shared = [self.shared_encoder(features) for features in projected]
        specific = [
            encoder(features)
            for encoder, features in zip(self.specific_encoders, projected)
        ]
        reconstructed = [
            reconstructor(torch.cat([shared_part, specific_part], dim=-1))
            for reconstructor, shared_part, specific_part in zip(
                self.reconstructors, shared, specific
            )
        ]
        reconstructed_specific = [
            encoder(features)
            for encoder, features in zip(self.specific_encoders, reconstructed)
        ]

        shared_logits = [
            head(features) for head, features in zip(self.shared_heads, shared)
        ]
        specific_logits = [
            head(features)
            for head, features in zip(self.specific_heads, specific)
        ]
        consensus_logit = self.consensus_head(torch.cat(shared, dim=-1))

        modality_tokens = torch.stack([
            shared_part + specific_part
            for shared_part, specific_part in zip(shared, specific)
        ])
        attended_tokens, modality_attention = self.fusion_attention(
            modality_tokens,
            modality_tokens,
            modality_tokens,
            need_weights=True,
        )
        attended_tokens = self.fusion_norm(modality_tokens + attended_tokens)
        fusion_features = torch.cat(
            [*attended_tokens.unbind(dim=0), *shared, *specific], dim=-1
        )
        anchor_output = self.fusion_head(fusion_features)

        shared_sequences = [self._as_sequence(features) for features in shared]
        specific_sequences = [
            self._as_sequence(features) for features in specific
        ]
        ccdr_output = None
        if self.use_ccdr:
            ccdr_output = self.ccdr(shared_sequences, specific_sequences)
            modality_logits = torch.cat(shared_logits, dim=1)
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
            output = anchor_output + conflict_gate * torch.tanh(
                ccdr_output["conflict_delta"]
            )
        else:
            output = anchor_output

        result = {
            "origin_l": self._as_sequence(projected[0]),
            "origin_a": self._as_sequence(projected[1]),
            "origin_v": self._as_sequence(projected[2]),
            "s_l": specific_sequences[0],
            "s_a": specific_sequences[1],
            "s_v": specific_sequences[2],
            "c_l": shared_sequences[0],
            "c_a": shared_sequences[1],
            "c_v": shared_sequences[2],
            "s_l_r": reconstructed_specific[0].unsqueeze(-1),
            "s_a_r": reconstructed_specific[1].unsqueeze(-1),
            "s_v_r": reconstructed_specific[2].unsqueeze(-1),
            "recon_l": self._as_sequence(reconstructed[0]),
            "recon_a": self._as_sequence(reconstructed[1]),
            "recon_v": self._as_sequence(reconstructed[2]),
            "c_l_sim": shared[0],
            "c_a_sim": shared[1],
            "c_v_sim": shared[2],
            "logits_l_shared": shared_logits[0],
            "logits_a_shared": shared_logits[1],
            "logits_v_shared": shared_logits[2],
            "logits_l_hetero": specific_logits[0],
            "logits_a_hetero": specific_logits[1],
            "logits_v_hetero": specific_logits[2],
            "logits_c": consensus_logit,
            "output_logit": output,
            "modality_attention": modality_attention,
        }
        if ccdr_output is not None:
            result.update({
                "consensus_logit": ccdr_output["consensus_logit"],
                "support_delta": ccdr_output["support_delta"],
                "conflict_delta": ccdr_output["conflict_delta"],
                "conflict_energy": ccdr_output["conflict_energy"],
                "consensus_parts": ccdr_output["consensus_parts"],
                "support_parts": ccdr_output["support_parts"],
                "conflict_parts": ccdr_output["conflict_parts"],
                "route_gates": ccdr_output["route_gates"],
                "donor_weights": ccdr_output["donor_weights"],
                "agreement_gates": ccdr_output["agreement_gates"],
                "anchor_logit": anchor_output,
                "conflict_gate": conflict_gate,
                "support_scale": ccdr_output["support_scale"],
                "conflict_scale": ccdr_output["conflict_scale"],
                "branch_attentions": ccdr_output["branch_attentions"],
                "ccdr_reconstructed": ccdr_output["reconstructed_parts"],
                "ccdr_targets": ccdr_output["evidence_targets"],
                "evidence_attention": ccdr_output["evidence_attention"],
            })
        return result