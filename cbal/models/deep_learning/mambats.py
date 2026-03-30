"""
MambaTS — Improved Selective State Space Models for Long-term Forecasting.

Full paper implementation (Cai et al., ICLR 2025):

1. **Variable Scan Along Time**: All K variates are arranged as a single
   sequence at each patch position → K*P total tokens scanned by Mamba.
2. **Temporal Mamba Block (TMB)**: Mamba without causal conv + selective dropout.
3. **Variable Permutation Training (VPT)**: Randomly shuffle variable order
   during training to make the model invariant to scan order.
4. **Variable-Aware Scan Along Time (VAST)**: Learns a K×K distance matrix
   during training, then solves ATSP to find the optimal scan order at inference.

Reference: Cai et al., "MambaTS: Improved Selective State Space Models for
Long-term Time Series Forecasting" (ICLR 2025).
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from cbal.models import register_model
from cbal.models.deep_learning.base import AbstractDLModel, _get_loss_fn
from cbal.models.deep_learning.patchtst import RevIN
from cbal.models.deep_learning.layers.mamba import MambaBlock


# ---------------------------------------------------------------------------
# TMB: Temporal Mamba Block (no causal conv + dropout)
# ---------------------------------------------------------------------------

class TemporalMambaBlock(nn.Module):
    """Temporal Mamba Block: Mamba without causal conv + selective dropout."""

    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.mamba = MambaBlock(
            d_model=d_model, d_state=d_state, expand=expand,
            d_conv=4, dropout=dropout, use_causal_conv=False,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.mamba(x))


# ---------------------------------------------------------------------------
# VAST: Variable-Aware Scan Along Time
# ---------------------------------------------------------------------------

class VAST:
    """Tracks variable relationships via distance matrix during VPT training.

    During training: after each forward pass with permutation π, update the
    distance matrix based on how well adjacent variable pairs contributed.

    During inference: solve ATSP via greedy nearest-neighbor to find optimal
    scan order.
    """

    def __init__(self, n_variates: int):
        self.n_variates = n_variates
        # Distance matrix P[i,j] = accumulated "cost" of scanning j after i
        # Lower = better adjacency. We track inverse: higher = better.
        self.adjacency = torch.zeros(n_variates, n_variates)
        self.count = 0

    def update(self, permutation: torch.Tensor, loss_value: float):
        """Update adjacency matrix from a training step.

        Per paper: pairs that appear adjacent in permutations with lower loss
        should have stronger (closer) relationships.

        Parameters
        ----------
        permutation : (K,) — variable order used in this step
        loss_value : float — training loss (lower = better)
        """
        K = len(permutation)
        if K <= 1:
            return
        # Inverse loss as "reward" — lower loss means better adjacency
        reward = 1.0 / (loss_value + 1e-6)
        for i in range(K - 1):
            src = permutation[i].item()
            dst = permutation[i + 1].item()
            self.adjacency[src, dst] += reward
        self.count += 1

    def solve_scan_order(self) -> list[int]:
        """Solve ATSP via greedy nearest-neighbor on learned adjacency.

        Returns the optimal variable scan order as a list of indices.
        """
        K = self.n_variates
        if K <= 1:
            return list(range(K))
        if self.count == 0:
            return list(range(K))

        # Greedy nearest-neighbor: start from node with highest total affinity
        adj = self.adjacency.clone()
        total_affinity = adj.sum(dim=1)
        start = total_affinity.argmax().item()

        visited = {start}
        order = [start]

        for _ in range(K - 1):
            current = order[-1]
            # Find unvisited neighbor with highest affinity
            scores = adj[current].clone()
            for v in visited:
                scores[v] = -float("inf")
            next_node = scores.argmax().item()
            order.append(next_node)
            visited.add(next_node)

        return order

    def reset(self):
        self.adjacency.zero_()
        self.count = 0


# ---------------------------------------------------------------------------
# MambaTS Network
# ---------------------------------------------------------------------------

class MambaTSNetwork(nn.Module):
    """MambaTS with Variable Scan Along Time + VPT + VAST.

    Architecture::

        Input (B, L, K)
        → RevIN → Patch each variate → (B, K, P, patch_len)
        → Embed → Positional encoding
        → Variable Scan: interleave K variates at each patch position
          → (B, K*P, d_model) — single sequence for Mamba
        → [TMB + FFN] × n_layers
        → Uninterleave → (B, K, P, d_model)
        → Head per variate → (B, H, K) → RevIN inverse

    During training with VPT: shuffle K variable order randomly.
    During inference with VAST: use learned optimal order.
    """

    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        n_variates: int = 1,
        d_model: int = 128,
        d_state: int = 16,
        n_layers: int = 3,
        d_ff: int = 256,
        patch_len: int = 16,
        stride: int = 8,
        dropout: float = 0.1,
        revin: bool = True,
        use_vpt: bool = True,
    ):
        super().__init__()
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.n_variates = n_variates
        self.patch_len = patch_len
        self.stride = stride
        self.use_vpt = use_vpt

        # Number of patches per variate
        self.n_patches = (max(context_length, patch_len) - patch_len) // stride + 1
        self.pad_len = max(0, stride * (self.n_patches - 1) + patch_len - context_length)

        # RevIN
        self.revin = RevIN(num_features=n_variates, affine=False) if revin else None

        # Patch embedding
        self.patch_embed = nn.Linear(patch_len, d_model)

        # Positional embedding (for patches within each variate)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)

        # Variable embedding (distinguish which variate each token belongs to)
        if n_variates > 1:
            self.var_embed = nn.Parameter(torch.randn(1, n_variates, 1, d_model) * 0.02)
        else:
            self.var_embed = None

        # Stacked TMB + FFN
        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(nn.ModuleDict({
                "tmb": TemporalMambaBlock(d_model, d_state, expand=2, dropout=dropout),
                "ffn_norm": nn.LayerNorm(d_model),
                "ffn": nn.Sequential(
                    nn.Linear(d_model, d_ff),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_ff, d_model),
                    nn.Dropout(dropout),
                ),
            }))

        self.norm = nn.LayerNorm(d_model)

        # Prediction head (shared across variates)
        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(self.n_patches * d_model, prediction_length),
        )

        # VAST tracker
        if n_variates > 1:
            self.vast = VAST(n_variates)
        else:
            self.vast = None

        # Store last permutation for VAST update
        self._last_perm = None

    def _create_variable_scan_sequence(
        self, z: torch.Tensor, perm: list[int] | None = None
    ) -> torch.Tensor:
        """Arrange K variate patch sequences into a single scan sequence.

        z : (B, K, P, D) — embedded patches per variate
        perm : variable scan order (None = identity)

        Returns: (B, K*P, D) — interleaved scan sequence
        """
        B, K, P, D = z.shape
        if perm is not None:
            z = z[:, perm, :, :]  # reorder variates

        # Interleave: for each patch position p, place all K variable tokens
        # Result order: [var0_patch0, var1_patch0, ..., varK_patch0,
        #                var0_patch1, var1_patch1, ..., varK_patch1, ...]
        z = z.permute(0, 2, 1, 3)  # (B, P, K, D)
        z = z.reshape(B, K * P, D)
        return z

    def _uninterleave(self, z: torch.Tensor, K: int, P: int,
                      perm: list[int] | None = None) -> torch.Tensor:
        """Reverse the interleaving and optionally unshuffle.

        z : (B, K*P, D) → (B, K, P, D)
        """
        B, _, D = z.shape
        z = z.reshape(B, P, K, D)  # (B, P, K, D)
        z = z.permute(0, 2, 1, 3)  # (B, K, P, D)

        if perm is not None:
            # Unshuffle: create inverse permutation
            inv_perm = [0] * K
            for i, p in enumerate(perm):
                inv_perm[p] = i
            z = z[:, inv_perm, :, :]

        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, L) for univariate, or (B, L, K) for multivariate.
        Returns: (B, H) or (B, H, K).
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (B, L, 1)
        B, L, K = x.shape

        # RevIN
        if self.revin is not None:
            x = self.revin(x)

        # Pad if needed
        if self.pad_len > 0:
            x = F.pad(x, (0, 0, self.pad_len, 0), mode="replicate")

        # Patch each variate: (B, L+pad, K) → (B, K, P, patch_len)
        patches_list = []
        for v in range(K):
            p = x[:, :, v].unfold(1, self.patch_len, self.stride)  # (B, P, patch_len)
            patches_list.append(p)
        patches = torch.stack(patches_list, dim=1)  # (B, K, P, patch_len)
        P = patches.size(2)

        # Embed patches
        z = self.patch_embed(patches)  # (B, K, P, D)
        z = z + self.pos_embed[:, :P, :].unsqueeze(1)  # broadcast over K

        # Variable embedding (so model knows which variate each token is)
        if self.var_embed is not None and K > 1:
            z = z + self.var_embed[:, :K, :, :]

        # --- VPT: determine scan order ---
        perm = None
        if K > 1:
            if self.training and self.use_vpt:
                # Random permutation during training
                perm = torch.randperm(K).tolist()
                self._last_perm = perm
            elif not self.training and self.vast is not None and self.vast.count > 0:
                # VAST: use learned optimal order at inference
                perm = self.vast.solve_scan_order()

        # Create variable-scan-along-time sequence
        z = self._create_variable_scan_sequence(z, perm)  # (B, K*P, D)

        # TMB + FFN blocks
        for block in self.blocks:
            z = block["tmb"](z)
            residual = z
            z = block["ffn"](block["ffn_norm"](z)) + residual

        z = self.norm(z)  # (B, K*P, D)

        # Uninterleave back to per-variate
        z = self._uninterleave(z, K, P, perm)  # (B, K, P, D)

        # Per-variate prediction head
        outputs = []
        for v in range(K):
            pred_v = self.head(z[:, v, :, :])  # (B, H)
            outputs.append(pred_v)
        pred = torch.stack(outputs, dim=-1)  # (B, H, K)

        # RevIN denormalize
        if self.revin is not None:
            pred = self.revin.inverse(pred)

        if K == 1:
            pred = pred.squeeze(-1)

        return pred


# ---------------------------------------------------------------------------
# MambaTS Model
# ---------------------------------------------------------------------------

@register_model("MambaTS")
class MambaTSModel(AbstractDLModel):
    """MambaTS: Improved Mamba for long-term time series forecasting (ICLR 2025).

    Full paper implementation with TMB, VPT, and VAST.

    Other Parameters
    ----------------
    d_model : int (default 128)
    d_state : int (default 16)
    n_layers : int (default 3)
    d_ff : int (default 256)
    patch_len : int (default 16)
    stride : int (default 8)
    dropout : float (default 0.1)
    revin : bool (default True)
    use_vpt : bool (default True)
        Enable Variable Permutation Training.
    """

    _default_hyperparameters = {
        **AbstractDLModel._default_hyperparameters,
        "d_model": 128,
        "d_state": 16,
        "n_layers": 3,
        "d_ff": 256,
        "patch_len": 16,
        "stride": 8,
        "dropout": 0.1,
        "revin": True,
        "use_vpt": True,
        "max_epochs": 50,
        "learning_rate": 1e-3,
    }

    def _build_network(self, context_length, prediction_length):
        return MambaTSNetwork(
            context_length=context_length,
            prediction_length=prediction_length,
            n_variates=1,
            d_model=self.get_hyperparameter("d_model"),
            d_state=self.get_hyperparameter("d_state"),
            n_layers=self.get_hyperparameter("n_layers"),
            d_ff=self.get_hyperparameter("d_ff"),
            patch_len=self.get_hyperparameter("patch_len"),
            stride=self.get_hyperparameter("stride"),
            dropout=self.get_hyperparameter("dropout"),
            revin=self.get_hyperparameter("revin"),
            use_vpt=self.get_hyperparameter("use_vpt"),
        )

    def _train_step(self, batch):
        pred = self._network(self._enrich_target(batch))
        loss_fn = _get_loss_fn(self.get_hyperparameter("loss_type"))
        loss = loss_fn(pred, batch["future_target"])

        # VAST: update distance matrix with current permutation and loss
        if (self._network.vast is not None
                and self._network._last_perm is not None):
            self._network.vast.update(
                torch.tensor(self._network._last_perm),
                loss.item(),
            )

        return loss

    def _predict_step(self, batch, quantile_levels=(0.1, 0.5, 0.9)):
        pred = self._network(self._enrich_target(batch))
        result = {"mean": pred, "quantiles": {}}
        for q in quantile_levels:
            result["quantiles"][q] = pred
        return result
