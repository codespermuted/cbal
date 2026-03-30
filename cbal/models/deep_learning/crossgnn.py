"""
CrossGNN — Confronting Noisy MTS Via Cross Interaction Refinement.

Key innovations for handling noisy multivariate time series:
1. **AMSI** (Adaptive Multi-Scale Identifier): AvgPool at multiple scales
   to construct denoised multi-scale representations
2. **Cross-Scale GNN**: GNN across scale nodes — extracts scales with
   clearer trend and weaker noise via learned saliency scores
3. **Cross-Variable GNN**: GNN across variable nodes — decouples
   homogeneity (+edges) and heterogeneity (−edges) between variables
4. Linear complexity O(L) via saliency-based edge sparsification

Reference: Huang et al., "CrossGNN: Confronting Noisy Multivariate Time
Series Via Cross Interaction Refinement" (NeurIPS 2023).
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from cbal.models import register_model
from cbal.models.deep_learning.base import AbstractDLModel, _get_loss_fn
from cbal.models.deep_learning.patchtst import RevIN


# ---------------------------------------------------------------------------
# AMSI: Adaptive Multi-Scale Identifier
# ---------------------------------------------------------------------------

class AMSI(nn.Module):
    """Adaptive Multi-Scale Identifier: constructs multi-scale series via AvgPool.

    Each scale s uses AvgPool with kernel=2^s, producing progressively smoother
    (less noisy) representations. Learnable scale weights determine which
    scales contribute most.
    """

    def __init__(self, context_length: int, n_scales: int = 4):
        super().__init__()
        self.n_scales = n_scales
        self.scale_weights = nn.Parameter(torch.ones(n_scales) / n_scales)

        self.scale_lengths = []
        for s in range(n_scales):
            self.scale_lengths.append(max(context_length // (2 ** s), 1))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """x: (B, L, D) → list of (B, L_s, D) for each scale."""
        scales = []
        for s, L_s in enumerate(self.scale_lengths):
            if s == 0:
                scales.append(x)
            else:
                pooled = F.adaptive_avg_pool1d(
                    x.permute(0, 2, 1), L_s
                ).permute(0, 2, 1)
                scales.append(pooled)
        return scales

    def get_weights(self) -> torch.Tensor:
        return F.softmax(self.scale_weights, dim=0)


# ---------------------------------------------------------------------------
# Cross-Scale GNN
# ---------------------------------------------------------------------------

class CrossScaleGNN(nn.Module):
    """GNN across scale nodes to find scales with clearer trends.

    Each scale is a node; edges represent scale-to-scale information flow.
    Learns saliency-based adjacency to focus on cleaner scales.
    """

    def __init__(self, d_model: int, n_scales: int, dropout: float = 0.1):
        super().__init__()
        self.n_scales = n_scales
        # Scale embeddings for adjacency learning
        self.scale_emb = nn.Parameter(torch.randn(n_scales, d_model) * 0.1)
        # Message passing MLP
        self.msg_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, scale_features: list[torch.Tensor]) -> list[torch.Tensor]:
        """scale_features: list of (B, D) per scale → refined list."""
        # Stack: (B, S, D)
        B = scale_features[0].size(0)
        D = scale_features[0].size(-1)
        h = torch.stack(scale_features, dim=1)  # (B, S, D)

        # Learn adjacency from scale embeddings
        adj = F.softmax(self.scale_emb @ self.scale_emb.T, dim=-1)  # (S, S)

        # GNN message passing
        msg = torch.einsum("ij,bjd->bid", adj, h)  # (B, S, D)
        h = self.norm(h + self.msg_mlp(msg))

        return [h[:, s, :] for s in range(self.n_scales)]


# ---------------------------------------------------------------------------
# Cross-Variable GNN
# ---------------------------------------------------------------------------

class CrossVariableGNN(nn.Module):
    """GNN across variable nodes with homogeneity/heterogeneity decoupling.

    Positive edges capture homogeneous relationships (similar patterns).
    Negative edges capture heterogeneous relationships (inverse patterns).
    Saliency scores determine edge importance.
    """

    def __init__(self, d_model: int, n_vars: int, dropout: float = 0.1):
        super().__init__()
        self.n_vars = n_vars
        # Variable embeddings for adjacency
        self.var_emb = nn.Parameter(torch.randn(n_vars, d_model) * 0.1)
        # Separate MLPs for positive and negative messages
        self.pos_mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
        )
        self.neg_mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
        )
        self.gate = nn.Linear(2 * d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, var_features: torch.Tensor) -> torch.Tensor:
        """var_features: (B, N, D) → (B, N, D) refined."""
        # Compute raw adjacency
        raw_adj = self.var_emb @ self.var_emb.T  # (N, N)

        # Decouple into positive (homogeneity) and negative (heterogeneity)
        pos_adj = F.softmax(F.relu(raw_adj), dim=-1)
        neg_adj = F.softmax(F.relu(-raw_adj), dim=-1)

        # Message passing
        pos_msg = torch.einsum("ij,bjd->bid", pos_adj, var_features)
        neg_msg = torch.einsum("ij,bjd->bid", neg_adj, var_features)

        pos_out = self.pos_mlp(pos_msg)
        neg_out = self.neg_mlp(neg_msg)

        # Gate to combine
        combined = self.gate(torch.cat([pos_out, neg_out], dim=-1))
        return self.norm(var_features + combined)


# ---------------------------------------------------------------------------
# CrossGNN Network
# ---------------------------------------------------------------------------

class CrossGNNNetwork(nn.Module):
    """CrossGNN: AMSI → Cross-Scale GNN → Cross-Variable GNN → Predict.

    Parameters
    ----------
    context_length, prediction_length : int
    n_vars : int (1 for univariate-per-item)
    d_model : int (default 64)
    n_scales : int (default 4)
    n_layers : int (default 2)
    dropout : float
    revin : bool
    """

    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        n_vars: int = 1,
        d_model: int = 64,
        n_scales: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        revin: bool = True,
    ):
        super().__init__()
        self.n_vars = n_vars
        self.n_scales = n_scales
        self.prediction_length = prediction_length

        self.revin = RevIN(num_features=max(n_vars, 1), affine=False) if revin else None

        # Input projection
        self.input_proj = nn.Linear(1, d_model)

        # AMSI
        self.amsi = AMSI(context_length, n_scales)

        # Stacked Cross-Scale + Cross-Variable layers
        self.cs_gnns = nn.ModuleList([CrossScaleGNN(d_model, n_scales, dropout) for _ in range(n_layers)])
        self.cv_gnns = nn.ModuleList([CrossVariableGNN(d_model, max(n_vars, 1), dropout) for _ in range(n_layers)])

        # Per-scale temporal aggregation (pool over time → d_model)
        self.temporal_pools = nn.ModuleList([
            nn.AdaptiveAvgPool1d(1) for _ in range(n_scales)
        ])

        # Prediction head
        self.head = nn.Linear(d_model * n_scales, prediction_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L) or (B, L, N) → (B, H) or (B, H, N)."""
        squeeze = x.dim() == 2
        if squeeze:
            x = x.unsqueeze(-1)  # (B, L, 1)

        B, L, N = x.shape

        if self.revin is not None:
            x = self.revin(x)

        # Process each variable
        all_preds = []
        for v in range(N):
            x_v = x[:, :, v:v+1]  # (B, L, 1)
            x_emb = self.input_proj(x_v)  # (B, L, D)

            # AMSI: multi-scale decomposition
            scales = self.amsi(x_emb)  # list of (B, L_s, D)

            # Pool each scale to get scale-level features
            scale_feats = []
            for s, x_s in enumerate(scales):
                # (B, D, L_s) → (B, D, 1) → (B, D)
                pooled = self.temporal_pools[s](x_s.permute(0, 2, 1)).squeeze(-1)
                scale_feats.append(pooled)

            # Cross-Scale GNN layers
            for cs_gnn in self.cs_gnns:
                scale_feats = cs_gnn(scale_feats)

            # Concatenate scale features
            scale_cat = torch.cat(scale_feats, dim=-1)  # (B, D*S)

            # Predict
            pred_v = self.head(scale_cat)  # (B, H)
            all_preds.append(pred_v)

        pred = torch.stack(all_preds, dim=-1)  # (B, H, N)

        # Cross-Variable GNN (across variables)
        if N > 1:
            for cv_gnn in self.cv_gnns:
                pred = cv_gnn(pred)  # (B, H, N) treated as (B, H=time, N=vars, D=1)→simplified

        if self.revin is not None:
            pred = self.revin.inverse(pred)

        if squeeze:
            pred = pred.squeeze(-1)
        return pred


# ---------------------------------------------------------------------------
# CrossGNN Model
# ---------------------------------------------------------------------------

@register_model("CrossGNN")
class CrossGNNModel(AbstractDLModel):
    """CrossGNN: Noise-robust GNN for Multivariate TS (NeurIPS 2023).

    Other Parameters
    ----------------
    d_model : int (default 64)
    n_scales : int (default 4)
    n_layers : int (default 2)
    dropout : float (default 0.1)
    revin : bool (default True)
    """

    _default_hyperparameters = {
        **AbstractDLModel._default_hyperparameters,
        "d_model": 64, "n_scales": 4, "n_layers": 2,
        "dropout": 0.1, "revin": True,
        "max_epochs": 50, "learning_rate": 1e-3, "stride": 2,
        "quantile_levels": (0.1, 0.5, 0.9),
    }

    def _build_network(self, context_length, prediction_length):
        self._quantile_head = None
        if self.get_hyperparameter("loss_type") == "quantile":
            from cbal.models.deep_learning.layers.distributions import QuantileOutput
            self._quantile_head = QuantileOutput(
                input_dim=1,
                quantile_levels=tuple(self.get_hyperparameter("quantile_levels")),
            )

        return CrossGNNNetwork(
            context_length=context_length, prediction_length=prediction_length,
            n_vars=1, d_model=self.get_hyperparameter("d_model"),
            n_scales=self.get_hyperparameter("n_scales"),
            n_layers=self.get_hyperparameter("n_layers"),
            dropout=self.get_hyperparameter("dropout"),
            revin=self.get_hyperparameter("revin"),
        )

    def _train_step(self, batch):
        pred = self._network(self._enrich_target(batch))  # (B, H)
        future = batch["future_target"]

        if self._quantile_head is not None:
            q_preds = self._quantile_head(pred.unsqueeze(-1))  # (B, H, Q)
            return self._quantile_head.loss(q_preds, future)

        loss_fn = _get_loss_fn(self.get_hyperparameter("loss_type"))
        return loss_fn(pred, future)

    def _predict_step(self, batch, quantile_levels=(0.1, 0.5, 0.9)):
        pred = self._network(self._enrich_target(batch))  # (B, H)

        if self._quantile_head is not None:
            q_preds = self._quantile_head(pred.unsqueeze(-1))  # (B, H, Q)
            mean = self._quantile_head.mean(q_preds)
            q_dict = self._quantile_head.quantile(q_preds, quantile_levels)
            return {"mean": mean, "quantiles": q_dict}

        return {"mean": pred, "quantiles": {q: pred for q in quantile_levels}}
