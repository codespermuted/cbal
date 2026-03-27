"""
SegRNN — Segment Recurrent Neural Network for Time Series Forecasting.

Instead of processing point-by-point (slow for long sequences), SegRNN
segments the input/output into fixed-length chunks. The RNN operates on
segments as tokens, reducing recurrence steps from L to L/seg_len.

Two decoding strategies:
- **RMR** (Recurrent Multi-step Regression): autoregressive — feed previous
  segment prediction as next decoder input. More expressive but slower.
- **PMR** (Parallel Multi-step Regression): use learned position embeddings
  as decoder input. Faster, no error accumulation.

Architecture::

    Encoder: Input (B, L) → segment (B, N_in, seg) → embed → GRU
    Decoder (RMR): last enc segment → embed → GRU step → project → feed back
    Decoder (PMR): learned queries → GRU (parallel) → project

Reference: Lin et al., "SegRNN: Segment Recurrent Neural Network for
Long-Term Time Series Forecasting" (arXiv 2023).
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from cbal.models import register_model
from cbal.models.deep_learning.base import AbstractDLModel
from cbal.models.deep_learning.patchtst import RevIN


class SegRNNNetwork(nn.Module):
    """SegRNN with proper segment-wise encoder-decoder.

    Parameters
    ----------
    context_length, prediction_length : int
    seg_len : int
        Segment length for both encoder and decoder (default 12).
    d_model : int
        GRU hidden dimension (default 128).
    n_layers : int
        GRU layers (default 1).
    dropout : float
    strategy : str
        ``"rmr"`` (recurrent) or ``"pmr"`` (parallel). Default ``"rmr"``.
    revin : bool
    """

    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        seg_len: int = 12,
        d_model: int = 128,
        n_layers: int = 1,
        dropout: float = 0.1,
        strategy: str = "rmr",
        revin: bool = True,
    ):
        super().__init__()
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.seg_len = seg_len
        self.d_model = d_model
        self.strategy = strategy.lower()
        assert self.strategy in ("rmr", "pmr"), f"Unknown strategy: {strategy}"

        # Number of encoder / decoder segments (ceiling division)
        self.n_seg_enc = max(1, (context_length + seg_len - 1) // seg_len)
        self.n_seg_dec = max(1, (prediction_length + seg_len - 1) // seg_len)

        # Padding to make encoder input divisible by seg_len
        self.pad_len = self.n_seg_enc * seg_len - context_length

        # RevIN
        self.revin = RevIN(num_features=1, affine=False) if revin else None

        # Encoder: segment embedding + GRU
        self.enc_embed = nn.Linear(seg_len, d_model)
        self.encoder = nn.GRU(
            input_size=d_model, hidden_size=d_model,
            num_layers=n_layers, batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )

        # Decoder: depends on strategy
        if self.strategy == "rmr":
            # RMR: autoregressive — feed previous segment prediction back
            self.dec_embed = nn.Linear(seg_len, d_model)
            self.decoder = nn.GRU(
                input_size=d_model, hidden_size=d_model,
                num_layers=n_layers, batch_first=True,
                dropout=dropout if n_layers > 1 else 0,
            )
        else:
            # PMR: parallel — learned position queries
            self.dec_queries = nn.Parameter(torch.randn(1, self.n_seg_dec, d_model) * 0.02)
            self.decoder = nn.GRU(
                input_size=d_model, hidden_size=d_model,
                num_layers=n_layers, batch_first=True,
                dropout=dropout if n_layers > 1 else 0,
            )

        # Segment output projection: d_model → seg_len
        self.seg_proj = nn.Linear(d_model, seg_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L) → (B, H)"""
        B = x.size(0)

        # RevIN
        if self.revin is not None:
            x = x.unsqueeze(-1)
            x = self.revin(x)
            x = x.squeeze(-1)

        # Pad left for encoder segmentation
        if self.pad_len > 0:
            x = F.pad(x, (self.pad_len, 0), mode="replicate")

        # ---- Encoder ----
        enc_segs = x.view(B, self.n_seg_enc, self.seg_len)        # (B, N_enc, seg)
        enc_emb = self.dropout(self.enc_embed(enc_segs))           # (B, N_enc, D)
        _, h = self.encoder(enc_emb)                                # h: (n_layers, B, D)

        # ---- Decoder ----
        if self.strategy == "rmr":
            pred = self._decode_rmr(B, h, enc_segs)
        else:
            pred = self._decode_pmr(B, h)

        # Trim to exact prediction_length
        pred = pred[:, :self.prediction_length]

        # RevIN denormalize
        if self.revin is not None:
            pred = pred.unsqueeze(-1)
            pred = self.revin.inverse(pred)
            pred = pred.squeeze(-1)

        return pred

    def _decode_rmr(self, B, h, enc_segs):
        """Recurrent Multi-step Regression: autoregressive segment decoding."""
        prev_seg = enc_segs[:, -1, :]  # (B, seg_len) — last encoder segment
        all_preds = []

        for _ in range(self.n_seg_dec):
            dec_input = self.dec_embed(prev_seg).unsqueeze(1)  # (B, 1, D)
            dec_out, h = self.decoder(dec_input, h)             # (B, 1, D)
            seg_pred = self.seg_proj(dec_out.squeeze(1))        # (B, seg_len)
            all_preds.append(seg_pred)
            prev_seg = seg_pred  # feed prediction back as next input

        return torch.cat(all_preds, dim=1)  # (B, n_seg_dec * seg_len)

    def _decode_pmr(self, B, h):
        """Parallel Multi-step Regression: all segments decoded at once."""
        queries = self.dec_queries.expand(B, -1, -1)  # (B, N_dec, D)
        dec_out, _ = self.decoder(queries, h)           # (B, N_dec, D)
        seg_preds = self.seg_proj(dec_out)              # (B, N_dec, seg_len)
        return seg_preds.reshape(B, -1)                 # (B, N_dec * seg_len)


@register_model("SegRNN")
class SegRNNModel(AbstractDLModel):
    """SegRNN: Segment-wise RNN for long-term time series forecasting.

    Other Parameters
    ----------------
    seg_len : int (default 12)
    d_model : int (default 128)
    n_layers : int (default 1)
    dropout : float (default 0.1)
    strategy : str
        ``"rmr"`` (recurrent, default) or ``"pmr"`` (parallel).
    revin : bool (default True)
    """

    _default_hyperparameters = {
        **AbstractDLModel._default_hyperparameters,
        "seg_len": 12, "d_model": 128, "n_layers": 1,
        "dropout": 0.1, "strategy": "rmr", "revin": True,
        "max_epochs": 50, "learning_rate": 1e-3, "stride": 2,
        "loss_type": "mse",            # "mse" or "quantile"
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

        return SegRNNNetwork(
            context_length=context_length, prediction_length=prediction_length,
            seg_len=self.get_hyperparameter("seg_len"),
            d_model=self.get_hyperparameter("d_model"),
            n_layers=self.get_hyperparameter("n_layers"),
            dropout=self.get_hyperparameter("dropout"),
            strategy=self.get_hyperparameter("strategy"),
            revin=self.get_hyperparameter("revin"),
        )

    def _train_step(self, batch):
        pred = self._network(self._enrich_target(batch))  # (B, H)
        future = batch["future_target"]

        if self._quantile_head is not None:
            q_preds = self._quantile_head(pred.unsqueeze(-1))  # (B, H, Q)
            return self._quantile_head.loss(q_preds, future)

        return F.mse_loss(pred, future)

    def _predict_step(self, batch, quantile_levels=(0.1, 0.5, 0.9)):
        pred = self._network(self._enrich_target(batch))  # (B, H)

        if self._quantile_head is not None:
            q_preds = self._quantile_head(pred.unsqueeze(-1))  # (B, H, Q)
            mean = self._quantile_head.mean(q_preds)
            q_dict = self._quantile_head.quantile(q_preds, quantile_levels)
            return {"mean": mean, "quantiles": q_dict}

        return {"mean": pred, "quantiles": {q: pred for q in quantile_levels}}
