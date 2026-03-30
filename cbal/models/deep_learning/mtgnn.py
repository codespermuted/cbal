"""
MTGNN — Connecting the Dots: Multivariate Time Series Forecasting with GNNs.

Core architecture:
1. **Graph Learning Layer**: learns directed adjacency matrix from learnable
   node embeddings E1, E2 via A = softmax(ReLU(E1·E2^T - E2·E1^T))
2. **Mix-hop Graph Convolution**: propagation with hop-mixing to avoid
   over-smoothing and retain local info
3. **Dilated Inception Temporal Conv**: multi-kernel dilated 1D conv for
   multi-scale temporal patterns
4. Interleaved temporal + graph conv layers with residual/skip connections

Reference: Wu et al., "Connecting the Dots: Multivariate Time Series
Forecasting with Graph Neural Networks" (KDD 2020).
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from cbal.models import register_model
from cbal.models.deep_learning.base import AbstractDLModel, _get_loss_fn


# ---------------------------------------------------------------------------
# Graph Learning Layer
# ---------------------------------------------------------------------------

class GraphLearning(nn.Module):
    """Learn a directed adjacency matrix from node embeddings.

    A = softmax(ReLU(α(E1·E2^T - E2·E1^T)))

    This discovers uni-directed relations between variables.
    """

    def __init__(self, n_nodes: int, embed_dim: int = 40, alpha: float = 3.0):
        super().__init__()
        self.alpha = alpha
        self.E1 = nn.Parameter(torch.randn(n_nodes, embed_dim) * 0.1)
        self.E2 = nn.Parameter(torch.randn(n_nodes, embed_dim) * 0.1)

    def forward(self) -> torch.Tensor:
        """Returns adjacency matrix (N, N)."""
        adj = F.relu(self.alpha * (self.E1 @ self.E2.T - self.E2 @ self.E1.T))
        return F.softmax(adj, dim=1)


# ---------------------------------------------------------------------------
# Mix-hop Graph Convolution
# ---------------------------------------------------------------------------

class MixhopGraphConv(nn.Module):
    """Mix-hop propagation: aggregates info from different hop distances.

    Avoids over-smoothing by concatenating features from each hop
    and mixing them with a learnable weight.
    """

    def __init__(self, d_in: int, d_out: int, n_hops: int = 2, dropout: float = 0.1):
        super().__init__()
        self.n_hops = n_hops
        # Linear for each hop (including hop 0 = self)
        self.linears = nn.ModuleList([nn.Linear(d_in, d_out) for _ in range(n_hops + 1)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x : (B, N, D_in) — node features
        adj : (N, N) — adjacency matrix
        Returns : (B, N, D_out)
        """
        out = self.linears[0](x)  # hop 0: self
        h = x
        for k in range(1, self.n_hops + 1):
            h = torch.einsum("ij,bjd->bid", adj, h)  # propagate
            out = out + self.linears[k](h)
        return self.dropout(out)


# ---------------------------------------------------------------------------
# Dilated Inception Temporal Conv
# ---------------------------------------------------------------------------

class DilatedInceptionConv(nn.Module):
    """Multi-kernel dilated 1D convolution (inception-style).

    Uses multiple kernel sizes [2, 3, 6, 7] with given dilation factor
    to capture multi-scale temporal patterns.
    """

    def __init__(self, d_in: int, d_out: int, dilation: int = 1,
                 kernel_sizes: tuple = (2, 3, 6, 7)):
        super().__init__()
        self.convs = nn.ModuleList()
        n_kernels = len(kernel_sizes)
        d_per_kernel = d_out // n_kernels
        remainder = d_out % n_kernels

        for i, k in enumerate(kernel_sizes):
            out_c = d_per_kernel + (1 if i < remainder else 0)
            padding = (k - 1) * dilation
            self.convs.append(nn.Conv1d(d_in, out_c, kernel_size=k,
                                         dilation=dilation, padding=padding))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B*N, D, L) → (B*N, D_out, L)"""
        outs = []
        for conv in self.convs:
            o = conv(x)[:, :, :x.size(2)]  # trim to original length
            outs.append(o)
        return torch.cat(outs, dim=1)


# ---------------------------------------------------------------------------
# MTGNN Layer (temporal conv + graph conv interleaved)
# ---------------------------------------------------------------------------

class MTGNNLayer(nn.Module):
    """Single MTGNN layer: temporal conv → graph conv with residual + skip."""

    def __init__(self, d_model: int, n_nodes: int, dilation: int = 1,
                 n_hops: int = 2, dropout: float = 0.1):
        super().__init__()
        # Temporal conv (filter + gate)
        self.filter_conv = DilatedInceptionConv(d_model, d_model, dilation)
        self.gate_conv = DilatedInceptionConv(d_model, d_model, dilation)

        # Graph conv (forward + backward)
        self.gc_forward = MixhopGraphConv(d_model, d_model, n_hops, dropout)
        self.gc_backward = MixhopGraphConv(d_model, d_model, n_hops, dropout)

        # Skip connection
        self.skip = nn.Conv1d(d_model, d_model, kernel_size=1)

        # Normalization
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        """
        x : (B, N, D, L) — batch × nodes × features × time
        adj : (N, N)
        Returns: x_out (B, N, D, L), skip (B, N, D, L)
        """
        B, N, D, L = x.shape
        residual = x

        # Temporal conv: reshape to (B*N, D, L)
        x_flat = x.reshape(B * N, D, L)
        x_filter = torch.tanh(self.filter_conv(x_flat))
        x_gate = torch.sigmoid(self.gate_conv(x_flat))
        x_tc = x_filter * x_gate  # (B*N, D, L)
        x_tc = self.dropout(x_tc)

        # Skip connection from temporal output
        skip = self.skip(x_tc).reshape(B, N, D, L)

        # Graph conv: reshape to (B, N, D*L) or process per time step
        # Efficient: process all time steps at once
        x_tc = x_tc.reshape(B, N, D, L)

        # Average over time for graph conv input, then broadcast back
        # More efficient: apply graph conv on (B, N, D) for each time step
        # We use a practical approach: permute to (B*L, N, D)
        x_perm = x_tc.permute(0, 3, 1, 2).reshape(B * L, N, D)
        gc_out = self.gc_forward(x_perm, adj) + self.gc_backward(x_perm, adj.T)
        gc_out = gc_out.reshape(B, L, N, D).permute(0, 2, 3, 1)  # (B, N, D, L)

        x_out = self.norm((residual + gc_out).permute(0, 1, 3, 2)).permute(0, 1, 3, 2)

        return x_out, skip


# ---------------------------------------------------------------------------
# MTGNN Network
# ---------------------------------------------------------------------------

class MTGNNNetwork(nn.Module):
    """Full MTGNN: graph learning + stacked temporal-graph layers.

    Parameters
    ----------
    n_nodes : int — number of variables (time series)
    context_length, prediction_length : int
    d_model : int (default 32)
    n_layers : int (default 3)
    embed_dim : int — node embedding dim for graph learning (default 10)
    n_hops : int — graph conv hops (default 2)
    dilation_base : int — dilation increases as base^layer (default 2)
    dropout : float
    """

    def __init__(
        self,
        n_nodes: int,
        context_length: int,
        prediction_length: int,
        d_model: int = 32,
        n_layers: int = 3,
        embed_dim: int = 10,
        n_hops: int = 2,
        dilation_base: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.prediction_length = prediction_length

        # Input projection: 1 → d_model per node
        self.input_proj = nn.Conv1d(1, d_model, kernel_size=1)

        # Graph learning
        self.graph_learn = GraphLearning(n_nodes, embed_dim)

        # Stacked layers with increasing dilation
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            dilation = dilation_base ** i
            self.layers.append(MTGNNLayer(d_model, n_nodes, dilation, n_hops, dropout))

        # Output: aggregate skips → project to prediction
        self.end_conv1 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.end_conv2 = nn.Conv1d(d_model, prediction_length, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, L, N) — multivariate input
        Returns : (B, H, N) — predictions for all nodes
        """
        B, L, N = x.shape

        # Learn adjacency
        adj = self.graph_learn()  # (N, N)

        # Input projection per node: (B, N, 1, L) → (B, N, D, L)
        x_nodes = []
        for n in range(N):
            x_n = self.input_proj(x[:, :, n].unsqueeze(1))  # (B, D, L)
            x_nodes.append(x_n)
        h = torch.stack(x_nodes, dim=1)  # (B, N, D, L)

        # Stacked layers
        skip_sum = 0
        for layer in self.layers:
            h, skip = layer(h, adj)
            skip_sum = skip_sum + skip

        # Output: aggregate across nodes
        # (B, N, D, L) → per node: (B, D, L) → (B, H)
        out = F.relu(skip_sum)
        outputs = []
        for n in range(N):
            o = out[:, n, :, :]  # (B, D, L)
            # Pool over time, then project
            o = F.relu(self.end_conv1(o))  # (B, D, L)
            o = self.end_conv2(o[:, :, -1:]).squeeze(-1)  # (B, H)
            outputs.append(o)

        return torch.stack(outputs, dim=-1)  # (B, H, N)


# ---------------------------------------------------------------------------
# MTGNN Model
# ---------------------------------------------------------------------------

@register_model("MTGNN")
class MTGNNModel(AbstractDLModel):
    """MTGNN: Graph Neural Network for Multivariate Time Series (KDD 2020).

    Automatically learns variable dependencies via graph learning.
    For univariate-per-item: N=1, graph conv is trivial.
    True power emerges with multivariate data.

    Other Parameters
    ----------------
    d_model : int (default 32)
    n_layers : int (default 3)
    embed_dim : int (default 10)
    n_hops : int (default 2)
    dropout : float (default 0.1)
    """

    _default_hyperparameters = {
        **AbstractDLModel._default_hyperparameters,
        "d_model": 32, "n_layers": 3, "embed_dim": 10,
        "n_hops": 2, "dropout": 0.1,
        "max_epochs": 50, "learning_rate": 1e-3,
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

        return MTGNNNetwork(
            n_nodes=1,  # univariate per item; multivariate extension overrides
            context_length=context_length,
            prediction_length=prediction_length,
            d_model=self.get_hyperparameter("d_model"),
            n_layers=self.get_hyperparameter("n_layers"),
            embed_dim=self.get_hyperparameter("embed_dim"),
            n_hops=self.get_hyperparameter("n_hops"),
            dropout=self.get_hyperparameter("dropout"),
        )

    def _train_step(self, batch):
        x = self._enrich_target(batch).unsqueeze(-1)  # (B, L, 1)
        pred = self._network(x).squeeze(-1)  # (B, H)
        future = batch["future_target"]

        if self._quantile_head is not None:
            q_preds = self._quantile_head(pred.unsqueeze(-1))  # (B, H, Q)
            return self._quantile_head.loss(q_preds, future)

        loss_fn = _get_loss_fn(self.get_hyperparameter("loss_type"))
        return loss_fn(pred, future)

    def _predict_step(self, batch, quantile_levels=(0.1, 0.5, 0.9)):
        x = self._enrich_target(batch).unsqueeze(-1)
        pred = self._network(x).squeeze(-1)  # (B, H)

        if self._quantile_head is not None:
            q_preds = self._quantile_head(pred.unsqueeze(-1))  # (B, H, Q)
            mean = self._quantile_head.mean(q_preds)
            q_dict = self._quantile_head.quantile(q_preds, quantile_levels)
            return {"mean": mean, "quantiles": q_dict}

        return {"mean": pred, "quantiles": {q: pred for q in quantile_levels}}
