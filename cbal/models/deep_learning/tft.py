"""
Temporal Fusion Transformer (TFT) — full paper implementation.

Architecture per Lim et al. (2021):

1. **Static Covariate Encoder**: entity embedding → 4 context vectors
   - c_s: context for Variable Selection Networks
   - c_e: context for static enrichment layer
   - c_h: LSTM hidden state initialization
   - c_c: LSTM cell state initialization

2. **3 separate Variable Selection Networks**:
   - Past observed VSN: target value (with c_s context)
   - Past known VSN: time features (with c_s context)
   - Future known VSN: time features (with c_s context)

3. **LSTM encoder-decoder**: initialized with (c_h, c_c) from static encoder

4. **Static enrichment**: GRN with c_e as context vector

5. **Interpretable Multi-Head Attention** + quantile output

Reference: Lim et al., "Temporal Fusion Transformers for Interpretable
Multi-horizon Time Series Forecasting" (Int. Journal of Forecasting, 2021).
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from cbal.models import register_model
from cbal.models.deep_learning.base import AbstractDLModel


# ============================================================================
# Building blocks (unchanged)
# ============================================================================

class GatedLinearUnit(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.fc = nn.Linear(d_in, d_out)
        self.gate = nn.Linear(d_in, d_out)

    def forward(self, x):
        return self.fc(x) * torch.sigmoid(self.gate(x))


class GRN(nn.Module):
    """Gated Residual Network."""

    def __init__(self, d_in, d_hidden, d_out=None, d_context=None, dropout=0.1):
        super().__init__()
        d_out = d_out or d_in
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)
        self.context_fc = nn.Linear(d_context, d_hidden, bias=False) if d_context else None
        self.glu = GatedLinearUnit(d_out, d_out)
        self.layernorm = nn.LayerNorm(d_out)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()

    def forward(self, x, context=None):
        residual = self.skip(x)
        h = self.fc1(x)
        if self.context_fc is not None and context is not None:
            # context: (B, d_context) → expand to match time dim if needed
            if context.dim() == 2 and x.dim() == 3:
                context = context.unsqueeze(1).expand(-1, x.size(1), -1)
            h = h + self.context_fc(context)
        h = F.elu(h)
        h = self.dropout(self.fc2(h))
        h = self.glu(h)
        return self.layernorm(residual + h)


class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network with static context."""

    def __init__(self, n_vars, d_model, d_input_per_var, d_context=None, dropout=0.1):
        super().__init__()
        self.n_vars = n_vars
        self.var_grns = nn.ModuleList([
            GRN(d_input_per_var, d_model, d_out=d_model, dropout=dropout)
            for _ in range(n_vars)
        ])
        self.selection_grn = GRN(
            d_in=n_vars * d_input_per_var, d_hidden=d_model, d_out=n_vars,
            d_context=d_context, dropout=dropout,
        )

    def forward(self, x_list, context=None):
        flat = torch.cat(x_list, dim=-1)
        ctx = context
        weights = F.softmax(self.selection_grn(flat, ctx), dim=-1)
        var_outputs = torch.stack(
            [grn(x_list[i]) for i, grn in enumerate(self.var_grns)], dim=-1,
        )
        return (var_outputs * weights.unsqueeze(2)).sum(dim=-1)


class InterpretableMultiHeadAttention(nn.Module):
    """Interpretable MHA — shared values across heads."""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, self.d_k)
        self.W_o = nn.Linear(self.d_k, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        B, T_q, _ = q.shape
        T_k = k.size(1)
        Q = self.W_q(q).view(B, T_q, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(B, T_k, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(v)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(scores, dim=-1))
        attn_avg = attn.mean(dim=1)
        out = torch.bmm(attn_avg, V)
        return self.W_o(out), attn_avg


class GateAddNorm(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.glu = GatedLinearUnit(d_model, d_model)
        self.layernorm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, residual):
        return self.layernorm(residual + self.dropout(self.glu(x)))


# ============================================================================
# Static Covariate Encoder (NEW — paper section 4.2)
# ============================================================================

class StaticCovariateEncoder(nn.Module):
    """Produces 4 context vectors from static (entity) embeddings.

    Per the paper, 4 separate GRNs produce:
    - c_s: context for Variable Selection
    - c_e: context for static enrichment
    - c_h: LSTM hidden state init
    - c_c: LSTM cell state init
    """

    def __init__(self, d_static: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.grn_cs = GRN(d_static, d_model, d_out=d_model, dropout=dropout)
        self.grn_ce = GRN(d_static, d_model, d_out=d_model, dropout=dropout)
        self.grn_ch = GRN(d_static, d_model, d_out=d_model, dropout=dropout)
        self.grn_cc = GRN(d_static, d_model, d_out=d_model, dropout=dropout)

    def forward(self, static_emb: torch.Tensor):
        """
        static_emb : (B, d_static)
        Returns: c_s, c_e, c_h, c_c — each (B, d_model)
        """
        c_s = self.grn_cs(static_emb)
        c_e = self.grn_ce(static_emb)
        c_h = self.grn_ch(static_emb)
        c_c = self.grn_cc(static_emb)
        return c_s, c_e, c_h, c_c


# ============================================================================
# TFT Network (rewritten per paper)
# ============================================================================

class TFTNetwork(nn.Module):
    """TFT with full paper architecture.

    Flow::

        Static: item_id → embedding → StaticCovariateEncoder → c_s, c_e, c_h, c_c
        Past: observed(target) → VSN_obs(c_s), known(time) → VSN_past_known(c_s)
              → combine → LSTM_enc(init=c_h,c_c) → gate
        Future: known(time) → VSN_future_known(c_s)
              → LSTM_dec → gate
        → Static Enrichment GRN(c_e) → Interpretable MHA → Quantile head
    """

    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_lstm_layers: int = 1,
        dropout: float = 0.1,
        n_time_features: int = 5,
        n_quantiles: int = 3,
        n_items: int = 1,
        embedding_dim: int = 16,
    ):
        super().__init__()
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.d_model = d_model
        self.n_quantiles = n_quantiles
        self.n_lstm_layers = n_lstm_layers

        # ---- Entity embedding (static categorical) ----
        self.entity_embedding = nn.Embedding(n_items, embedding_dim)

        # ---- Static Covariate Encoder ----
        self.static_encoder = StaticCovariateEncoder(embedding_dim, d_model, dropout)

        # ---- Input projections ----
        self.enc_target_proj = nn.Linear(1, d_model)       # past observed
        self.enc_time_proj = nn.Linear(n_time_features, d_model)  # past known
        self.dec_time_proj = nn.Linear(n_time_features, d_model)  # future known

        # ---- 3 separate VSNs (paper section 4.2) ----
        # Past observed: target value
        self.vsn_past_observed = VariableSelectionNetwork(
            n_vars=1, d_model=d_model, d_input_per_var=d_model,
            d_context=d_model, dropout=dropout,
        )
        # Past known: time features
        self.vsn_past_known = VariableSelectionNetwork(
            n_vars=1, d_model=d_model, d_input_per_var=d_model,
            d_context=d_model, dropout=dropout,
        )
        # Future known: time features
        self.vsn_future_known = VariableSelectionNetwork(
            n_vars=1, d_model=d_model, d_input_per_var=d_model,
            d_context=d_model, dropout=dropout,
        )

        # ---- Combine past observed + past known ----
        self.past_combine = GRN(2 * d_model, d_model, d_out=d_model, dropout=dropout)

        # ---- LSTM encoder-decoder ----
        self.lstm_encoder = nn.LSTM(
            d_model, d_model, num_layers=n_lstm_layers,
            batch_first=True, dropout=dropout if n_lstm_layers > 1 else 0,
        )
        self.lstm_decoder = nn.LSTM(
            d_model, d_model, num_layers=n_lstm_layers,
            batch_first=True, dropout=dropout if n_lstm_layers > 1 else 0,
        )

        # LSTM init projections (per layer)
        self.h_init_proj = nn.Linear(d_model, n_lstm_layers * d_model)
        self.c_init_proj = nn.Linear(d_model, n_lstm_layers * d_model)

        # Gate after LSTM
        self.enc_gate = GateAddNorm(d_model, dropout)
        self.dec_gate = GateAddNorm(d_model, dropout)

        # ---- Static enrichment (GRN with c_e context) ----
        self.enrichment_grn = GRN(d_model, d_model, d_context=d_model, dropout=dropout)

        # ---- Interpretable Multi-Head Attention ----
        self.attention = InterpretableMultiHeadAttention(d_model, n_heads, dropout)
        self.attn_gate = GateAddNorm(d_model, dropout)

        # ---- Positionwise feed-forward ----
        self.positionwise_grn = GRN(d_model, d_model, dropout=dropout)
        self.output_gate = GateAddNorm(d_model, dropout)

        # ---- Quantile output head ----
        self.output_proj = nn.Linear(d_model, n_quantiles)

    def forward(
        self,
        past_target: torch.Tensor,
        past_time_features: torch.Tensor,
        future_time_features: torch.Tensor,
        item_id_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        past_target : (B, C)
        past_time_features : (B, C, F)
        future_time_features : (B, H, F)
        item_id_index : (B,) long tensor — item indices for embedding

        Returns
        -------
        Tensor (B, H, n_quantiles)
        """
        B = past_target.size(0)
        device = past_target.device

        # ---- Static covariate encoder ----
        if item_id_index is not None:
            entity_emb = self.entity_embedding(item_id_index)  # (B, E)
        else:
            entity_emb = torch.zeros(B, self.entity_embedding.embedding_dim, device=device)

        c_s, c_e, c_h, c_c = self.static_encoder(entity_emb)  # each (B, D)

        # ---- LSTM initialization from static context ----
        h0 = self.h_init_proj(c_h).view(B, self.n_lstm_layers, self.d_model).permute(1, 0, 2).contiguous()
        c0 = self.c_init_proj(c_c).view(B, self.n_lstm_layers, self.d_model).permute(1, 0, 2).contiguous()

        # ---- Encoder: 3-way variable selection ----
        enc_target = self.enc_target_proj(past_target.unsqueeze(-1))  # (B, C, D)
        enc_time = self.enc_time_proj(past_time_features)              # (B, C, D)

        past_obs_selected = self.vsn_past_observed([enc_target], context=c_s)    # (B, C, D)
        past_known_selected = self.vsn_past_known([enc_time], context=c_s)       # (B, C, D)

        # Combine past observed + past known
        enc_input = self.past_combine(
            torch.cat([past_obs_selected, past_known_selected], dim=-1)
        )  # (B, C, D)

        # ---- LSTM encoder (initialized with c_h, c_c) ----
        enc_lstm_out, (h, c) = self.lstm_encoder(enc_input, (h0, c0))
        enc_out = self.enc_gate(enc_lstm_out, enc_input)  # (B, C, D)

        # ---- Decoder: future known VSN ----
        dec_time = self.dec_time_proj(future_time_features)  # (B, H, D)
        dec_selected = self.vsn_future_known([dec_time], context=c_s)  # (B, H, D)

        # ---- LSTM decoder ----
        dec_lstm_out, _ = self.lstm_decoder(dec_selected, (h, c))
        dec_out = self.dec_gate(dec_lstm_out, dec_selected)  # (B, H, D)

        # ---- Static enrichment (GRN with c_e) ----
        enriched = self.enrichment_grn(dec_out, context=c_e)  # (B, H, D)

        # ---- Interpretable Multi-Head Attention ----
        attn_out, _ = self.attention(q=enriched, k=enc_out, v=enc_out)
        attn_out = self.attn_gate(attn_out, enriched)

        # ---- Positionwise GRN ----
        pw_out = self.positionwise_grn(attn_out)
        final = self.output_gate(pw_out, attn_out)

        return self.output_proj(final)  # (B, H, Q)


# ============================================================================
# Loss + Model
# ============================================================================

class QuantileLoss(nn.Module):
    """Quantile loss (pinball loss) for multiple quantile levels."""

    def __init__(self, quantile_levels: list[float]):
        super().__init__()
        self.quantile_levels = quantile_levels

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.unsqueeze(-1)
        errors = target - pred
        losses = []
        for i, q in enumerate(self.quantile_levels):
            e = errors[..., i]
            losses.append(torch.max(q * e, (q - 1) * e))
        return torch.stack(losses, dim=-1).mean()


@register_model("TFT")
class TFTModel(AbstractDLModel):
    """Temporal Fusion Transformer — full paper implementation.

    Includes static covariate encoder (4 context vectors), 3 separate VSNs,
    entity embedding, and LSTM initialization from static context.

    Other Parameters
    ----------------
    d_model : int (default 64)
    n_heads : int (default 4)
    n_lstm_layers : int (default 1)
    dropout : float (default 0.1)
    embedding_dim : int (default 16)
    quantile_levels : list of float (default [0.1, 0.5, 0.9])
    """

    _default_hyperparameters = {
        **AbstractDLModel._default_hyperparameters,
        "d_model": 64,
        "n_heads": 4,
        "n_lstm_layers": 1,
        "dropout": 0.1,
        "embedding_dim": 16,
        "quantile_levels": [0.1, 0.5, 0.9],
        "max_epochs": 100,
        "learning_rate": 1e-3,
    }

    def _fit(self, train_data, val_data=None, time_limit=None):
        self._quantile_levels = self.get_hyperparameter("quantile_levels")
        self._loss_fn = QuantileLoss(self._quantile_levels)
        # Item mapping (shared with dataset)
        self._item_id_to_idx = {
            item_id: idx for idx, item_id in enumerate(sorted(train_data.item_ids))
        }
        self._n_items = len(self._item_id_to_idx)
        super()._fit(train_data, val_data, time_limit)

    def _build_network(self, context_length, prediction_length):
        return TFTNetwork(
            context_length=context_length,
            prediction_length=prediction_length,
            d_model=self.get_hyperparameter("d_model"),
            n_heads=self.get_hyperparameter("n_heads"),
            n_lstm_layers=self.get_hyperparameter("n_lstm_layers"),
            dropout=self.get_hyperparameter("dropout"),
            n_time_features=5,
            n_quantiles=len(self._quantile_levels),
            n_items=self._n_items,
            embedding_dim=self.get_hyperparameter("embedding_dim"),
        )

    def _train_step(self, batch):
        pred = self._network(
            past_target=batch["past_target"],
            past_time_features=batch["past_time_features"],
            future_time_features=batch["future_time_features"],
            item_id_index=batch.get("item_id_index"),
        )
        return self._loss_fn(pred, batch["future_target"])

    def _predict_step(self, batch, quantile_levels=(0.1, 0.5, 0.9)):
        pred = self._network(
            past_target=batch["past_target"],
            past_time_features=batch["past_time_features"],
            future_time_features=batch["future_time_features"],
            item_id_index=batch.get("item_id_index"),
        )

        result = {"quantiles": {}}
        median_idx = None
        for i, q in enumerate(self._quantile_levels):
            result["quantiles"][q] = pred[:, :, i]
            if abs(q - 0.5) < 0.01:
                median_idx = i

        result["mean"] = pred[:, :, median_idx] if median_idx is not None else pred.mean(dim=-1)
        return result
