"""
AbstractDLModel — Base class for all PyTorch-based deep learning models.

Handles:
- Device management (CPU/GPU auto-detection)
- Training loop with gradient clipping
- Early stopping on validation loss
- Learning rate scheduling
- Checkpoint saving/loading
- Standardized prediction pipeline

Subclasses implement:
- ``_build_network()`` — construct the ``nn.Module``
- ``_train_step()`` — one forward pass + loss computation
- ``_predict_step()`` — generate forecasts from a batch
"""

from __future__ import annotations

import abc
import logging
import time
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cbal.dataset.ts_dataframe import ITEMID, TIMESTAMP, TimeSeriesDataFrame
from cbal.models.abstract_model import AbstractTimeSeriesModel
from cbal.models.deep_learning.dataset import TimeSeriesDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loss function factory — align training loss with eval metric
# ---------------------------------------------------------------------------

def _get_loss_fn(loss_type: str):
    """Return a loss function matching the given type.

    Supported: "mse", "mae", "huber", "quantile" (handled separately by models).
    Using MAE loss when eval_metric=MAE directly optimizes the target metric
    instead of the proxy MSE.
    """
    import torch.nn.functional as F

    if loss_type == "mae":
        return F.l1_loss
    elif loss_type == "huber":
        return F.smooth_l1_loss
    elif loss_type == "mse":
        return F.mse_loss
    else:
        # "quantile" and others — return MSE as fallback (quantile handled in model)
        return F.mse_loss


class _WarmupScheduler:
    """Linear warmup followed by another scheduler.

    During the first ``warmup_epochs``, the LR linearly increases from
    a small value to ``base_lr``.  After warmup, delegates to
    ``after_scheduler`` (e.g. CosineAnnealingLR).
    """

    def __init__(self, optimizer, after_scheduler, warmup_epochs, base_lr):
        self._optimizer = optimizer
        self._after = after_scheduler
        self._warmup_epochs = warmup_epochs
        self._base_lr = base_lr
        self._current_epoch = 0

    def step(self):
        self._current_epoch += 1
        if self._current_epoch <= self._warmup_epochs:
            # Linear warmup: lr goes from base_lr * 0.01 to base_lr
            frac = self._current_epoch / self._warmup_epochs
            lr = self._base_lr * (0.01 + 0.99 * frac)
            for pg in self._optimizer.param_groups:
                pg["lr"] = lr
        else:
            self._after.step()


class AbstractDLModel(AbstractTimeSeriesModel):
    """Base class for deep learning forecasting models.

    All DL models share:
    - Common hyperparameters (lr, epochs, batch_size, etc.)
    - Training loop with EarlyStopping
    - Prediction pipeline
    - Device management

    Subclasses must implement:
    - ``_build_network(context_length, prediction_length, **kwargs) -> nn.Module``
    - ``_train_step(batch) -> loss_tensor``
    - ``_predict_step(batch) -> dict[str, Tensor]``

    Other Parameters
    ----------------
    context_length : int
        Number of past time steps to use as input. Default: 3 * prediction_length.
    max_epochs : int
        Maximum number of training epochs. Default: 100.
    batch_size : int
        Training batch size. Default: 32.
    learning_rate : float
        Initial learning rate. Default: 1e-3.
    patience : int
        Early stopping patience (epochs). Default: 10.
    grad_clip : float
        Gradient clipping norm. Default: 1.0.
    num_workers : int
        DataLoader workers. Default: 0.
    device : str or None
        ``"cpu"``, ``"cuda"``, or ``None`` (auto-detect).
    """

    _default_hyperparameters = {
        "context_length": None,  # None = 3 * prediction_length
        "max_epochs": 100,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "patience": 20,
        "grad_clip": 1.0,
        "num_workers": 0,
        "device": None,
        "val_fraction": 0.1,
        "stride": 1,
        # --- Optimizer & scheduler ---
        "optimizer": "nadam",          # "adam", "nadam", "adamw"
        "weight_decay": 1e-4,          # L2 regularization
        "warmup_fraction": 0.1,        # fraction of epochs for LR warmup
        "lr_scheduler": "cosine",      # "cosine", "plateau", "none"
        "lr_min_ratio": 0.01,          # min_lr = lr * lr_min_ratio
        "gradient_accumulation_steps": 1,
        "use_amp": True,               # automatic mixed precision (faster on GPU)
        "max_batches_per_epoch": 50,   # AG-style: cap batches per epoch for speed
        # --- Loss function ---
        "loss_type": "mae",            # "mse", "mae", "huber", "quantile"
                                       # MAE aligns with eval_metric=MAE for direct optimization
        # --- Per-item scaling (applied at base level) ---
        "target_scaling": "mean_abs",  # "mean_abs", "standard", "none"
    }

    # Subclasses that handle their own scaling should set this to True
    _uses_own_scaling: bool = False

    def _fit(self, train_data, val_data=None, time_limit=None):
        # Resolve hyperparameters
        ctx_len = self.get_hyperparameter("context_length")
        if ctx_len is None:
            ctx_len = max(3 * self.prediction_length, 30)
        self._context_length = ctx_len
        self._device = self._resolve_device()

        # --- Per-item scaling (AutoGluon-style) ---
        # For models with RevIN: skip base scaling (RevIN handles per-window normalization)
        # Double normalization (mean_abs + RevIN) hurts non-stationary data severely
        scaling_method = self.get_hyperparameter("target_scaling")
        has_revin = self.get_hyperparameter("revin") if "revin" in self._hyperparameters else False
        if has_revin and scaling_method == "mean_abs":
            scaling_method = "none"  # RevIN handles normalization already
        self._apply_base_scaling = (
            scaling_method != "none" and not self._uses_own_scaling
        )
        self._item_scales: dict[str, tuple[float, float]] = {}  # item_id → (loc, scale)

        if self._apply_base_scaling:
            from cbal.dataset.ts_dataframe import TARGET
            train_data = train_data.copy()
            for item_id in train_data.item_ids:
                y = train_data.loc[item_id][TARGET].values.astype(np.float64)
                y_finite = y[np.isfinite(y)]
                if len(y_finite) == 0:
                    self._item_scales[item_id] = (0.0, 1.0)
                    continue

                if scaling_method == "mean_abs":
                    loc = 0.0
                    sc = float(max(np.mean(np.abs(y_finite)), 1e-8))
                elif scaling_method == "standard":
                    loc = float(np.mean(y_finite))
                    sc = float(max(np.std(y_finite), 1e-8))
                else:
                    loc, sc = 0.0, 1.0

                self._item_scales[item_id] = (loc, sc)
                mask = train_data.index.get_level_values(ITEMID) == item_id
                train_data.loc[mask, TARGET] = (
                    train_data.loc[mask, TARGET].values - loc
                ) / sc

            if val_data is not None:
                val_data = val_data.copy()
                for item_id in val_data.item_ids:
                    loc, sc = self._item_scales.get(item_id, (0.0, 1.0))
                    mask = val_data.index.get_level_values(ITEMID) == item_id
                    val_data.loc[mask, TARGET] = (
                        val_data.loc[mask, TARGET].values - loc
                    ) / sc

            logger.debug(f"  Base scaling applied: method={scaling_method}")

        # Build datasets
        # Subclasses (e.g. DeepAR) may set self._item_id_to_idx before calling super()._fit
        item_map = getattr(self, "_item_id_to_idx", None)

        train_ds = TimeSeriesDataset(
            train_data, ctx_len, self.prediction_length,
            freq=self.freq, mode="train",
            stride=self.get_hyperparameter("stride"),
            item_id_to_idx=item_map,
        )
        if len(train_ds) == 0:
            raise ValueError("No valid training windows. Increase data length or decrease context_length.")

        val_ds = None
        if val_data is not None:
            val_ds = TimeSeriesDataset(
                val_data, ctx_len, self.prediction_length,
                freq=self.freq, mode="train", stride=max(1, ctx_len // 2),
                item_id_to_idx=item_map,
            )

        batch_size = self.get_hyperparameter("batch_size")
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=self.get_hyperparameter("num_workers"),
            drop_last=len(train_ds) > batch_size,
        )
        val_loader = None
        if val_ds is not None and len(val_ds) > 0:
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        # Detect covariate dimensions from first batch
        sample_batch = next(iter(train_loader))
        self._n_past_features = (
            sample_batch["past_time_features"].shape[-1]
            if "past_time_features" in sample_batch
            and sample_batch["past_time_features"].dim() >= 2
            else 0
        )

        # Build network
        self._network = self._build_network(
            context_length=ctx_len,
            prediction_length=self.prediction_length,
        ).to(self._device)

        # Ensure all network submodules are on the correct device
        self._network = self._network.to(self._device)

        # Move any auxiliary modules (e.g., quantile_head) created in _build_network
        for attr_name in dir(self):
            if attr_name.startswith("_") and not attr_name.startswith("__"):
                attr = getattr(self, attr_name, None)
                if isinstance(attr, nn.Module):
                    setattr(self, attr_name, attr.to(self._device))

        # Additive covariate injection: enriched = target + gate * MLP(covariates)
        # The target signal is NEVER modified by the projection — covariates
        # are added as a learned residual. Gate starts at 0 so training begins
        # from pure target, then gradually incorporates covariate info.
        self._cov_projector = None
        self._cov_gate = None
        if self._n_past_features > 0:
            self._cov_projector = nn.Sequential(
                nn.Linear(self._n_past_features, max(self._n_past_features // 2, 4)),
                nn.GELU(),
                nn.Linear(max(self._n_past_features // 2, 4), 1),
            ).to(self._device)
            # Learnable gate initialized to 0 → starts as pure target
            self._cov_gate = nn.Parameter(
                torch.zeros(1, device=self._device)
            )

        # --- Optimizer ---
        lr = self.get_hyperparameter("learning_rate")
        weight_decay = self.get_hyperparameter("weight_decay")
        opt_name = self.get_hyperparameter("optimizer").lower()

        # Collect all parameters (network + covariate injector + any auxiliary modules)
        all_params = list(self._network.parameters())
        if self._cov_projector is not None:
            all_params += list(self._cov_projector.parameters())
        if self._cov_gate is not None:
            all_params += [self._cov_gate]
        # Include auxiliary modules (e.g., quantile_head) set by _build_network
        for attr_name in dir(self):
            if attr_name.startswith("_") and not attr_name.startswith("__"):
                attr = getattr(self, attr_name, None)
                if isinstance(attr, nn.Module) and attr is not self._network and attr is not self._cov_projector:
                    all_params += list(attr.parameters())

        if opt_name == "nadam":
            optimizer = torch.optim.NAdam(
                all_params, lr=lr, weight_decay=weight_decay,
            )
        elif opt_name == "adamw":
            optimizer = torch.optim.AdamW(
                all_params, lr=lr, weight_decay=weight_decay,
            )
        else:  # "adam"
            optimizer = torch.optim.Adam(
                all_params, lr=lr, weight_decay=weight_decay,
            )

        # --- LR Scheduler ---
        max_epochs = self.get_hyperparameter("max_epochs")
        warmup_fraction = self.get_hyperparameter("warmup_fraction")
        lr_sched_name = self.get_hyperparameter("lr_scheduler").lower()
        lr_min_ratio = self.get_hyperparameter("lr_min_ratio")
        warmup_epochs = max(int(max_epochs * warmup_fraction), 1)

        if lr_sched_name == "cosine":
            # Cosine annealing after warmup
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(max_epochs - warmup_epochs, 1),
                eta_min=lr * lr_min_ratio,
            )
            scheduler = _WarmupScheduler(optimizer, cosine_scheduler, warmup_epochs, lr)
        elif lr_sched_name == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5, min_lr=lr * lr_min_ratio,
            )
        else:
            scheduler = None

        # --- Mixed precision ---
        use_amp = self.get_hyperparameter("use_amp") and self._device.type == "cuda"
        try:
            scaler = torch.amp.GradScaler(self._device.type) if use_amp else None
        except Exception:
            use_amp = False
            scaler = None

        # Training loop
        patience = self.get_hyperparameter("patience")
        grad_clip = self.get_hyperparameter("grad_clip")
        accum_steps = self.get_hyperparameter("gradient_accumulation_steps")
        max_batches = self.get_hyperparameter("max_batches_per_epoch")
        best_val_loss = float("inf")
        best_state = None
        epochs_no_improve = 0
        start_time = time.time()

        for epoch in range(max_epochs):
            # Time limit check
            if time_limit is not None and (time.time() - start_time) > time_limit * 0.9:
                logger.info(f"  Time limit reached at epoch {epoch}")
                break

            # Train
            self._network.train()
            train_loss_sum = 0.0
            n_batches = 0
            optimizer.zero_grad()

            for step, batch in enumerate(train_loader):
                # AG-style: cap batches per epoch for speed on large datasets
                if max_batches and step >= max_batches:
                    break
                batch = self._to_device(batch)

                try:
                    if use_amp and scaler is not None:
                        with torch.amp.autocast(self._device.type):
                            loss = self._train_step(batch)
                        loss = loss / accum_steps
                        scaler.scale(loss).backward()
                        if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                            if grad_clip > 0:
                                scaler.unscale_(optimizer)
                                nn.utils.clip_grad_norm_(self._network.parameters(), grad_clip)
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()
                    else:
                        loss = self._train_step(batch)
                        loss = loss / accum_steps
                        loss.backward()
                        if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                            if grad_clip > 0:
                                nn.utils.clip_grad_norm_(self._network.parameters(), grad_clip)
                            optimizer.step()
                            optimizer.zero_grad()
                except RuntimeError as _amp_err:
                    # AMP can fail on some architectures; fall back to fp32
                    if use_amp and "inf checks" in str(_amp_err):
                        use_amp = False
                        scaler = None
                        loss = self._train_step(batch)
                        loss = loss / accum_steps
                        loss.backward()
                        if grad_clip > 0:
                            nn.utils.clip_grad_norm_(self._network.parameters(), grad_clip)
                        optimizer.step()
                        optimizer.zero_grad()
                    else:
                        raise

                train_loss_sum += loss.item() * accum_steps
                n_batches += 1

            avg_train_loss = train_loss_sum / max(n_batches, 1)

            # Validate
            avg_val_loss = avg_train_loss  # fallback
            if val_loader is not None:
                avg_val_loss = self._evaluate(val_loader)

            # Step scheduler
            if scheduler is not None:
                if lr_sched_name == "plateau":
                    scheduler.step(avg_val_loss)
                else:
                    scheduler.step()

            # Early stopping
            if avg_val_loss < best_val_loss - 1e-6:
                best_val_loss = avg_val_loss
                best_state = {k: v.cpu().clone() for k, v in self._network.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epoch % 10 == 0 or epochs_no_improve == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                logger.debug(
                    f"  Epoch {epoch:3d} | train_loss={avg_train_loss:.4f} "
                    f"val_loss={avg_val_loss:.4f} | best={best_val_loss:.4f} "
                    f"lr={current_lr:.2e}"
                )

            if epochs_no_improve >= patience:
                logger.info(f"  Early stopping at epoch {epoch}")
                break

        # Restore best weights and ensure on correct device
        if best_state is not None:
            self._network.load_state_dict(best_state)
        self._network = self._network.to(self._device)
        self._network.eval()

    def _predict(self, data, known_covariates=None, quantile_levels=None):
        quantile_levels = quantile_levels or [0.1, 0.5, 0.9]

        # Scale input data if base scaling was used during training
        predict_data = data
        if getattr(self, "_apply_base_scaling", False) and self._item_scales:
            from cbal.dataset.ts_dataframe import TARGET
            predict_data = data.copy()
            for item_id in predict_data.item_ids:
                loc, sc = self._item_scales.get(item_id, (0.0, 1.0))
                mask = predict_data.index.get_level_values(ITEMID) == item_id
                predict_data.loc[mask, TARGET] = (
                    predict_data.loc[mask, TARGET].values - loc
                ) / sc

        item_map = getattr(self, "_item_id_to_idx", None)
        pred_ds = TimeSeriesDataset(
            predict_data, self._context_length, self.prediction_length,
            freq=self.freq, mode="predict",
            item_id_to_idx=item_map,
        )
        pred_loader = DataLoader(
            pred_ds, batch_size=self.get_hyperparameter("batch_size"), shuffle=False,
        )

        self._network = self._network.to(self._device)
        self._network.eval()
        all_rows = []

        with torch.no_grad():
            for batch in pred_loader:
                batch = self._to_device(batch)
                result = self._predict_step(batch, quantile_levels=quantile_levels)

                # result: {item_ids: list[str], mean: (B, H), quantiles: {q: (B, H)}}
                item_ids = batch["item_id"]
                B = len(item_ids)

                for b in range(B):
                    item_id = item_ids[b]
                    future_ts = self._make_future_timestamps(data, item_id)

                    # Inverse scale if base scaling was applied
                    loc, sc = (0.0, 1.0)
                    if getattr(self, "_apply_base_scaling", False):
                        loc, sc = self._item_scales.get(item_id, (0.0, 1.0))

                    for h in range(self.prediction_length):
                        raw_mean = result["mean"][b, h].cpu().item()
                        row = {
                            ITEMID: item_id,
                            TIMESTAMP: future_ts[h],
                            "mean": raw_mean * sc + loc,
                        }
                        for q in quantile_levels:
                            raw_q = result["quantiles"][q][b, h].cpu().item()
                            row[str(q)] = raw_q * sc + loc
                        all_rows.append(row)

        return self._rows_to_tsdf(all_rows)

    # ------------------------------------------------------------------
    # Abstract methods for subclasses
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def _build_network(self, context_length: int, prediction_length: int) -> nn.Module:
        """Build and return the PyTorch model."""
        ...

    @abc.abstractmethod
    def _train_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass + loss. Return scalar loss tensor."""
        ...

    @abc.abstractmethod
    def _predict_step(
        self, batch: dict[str, torch.Tensor], quantile_levels: Sequence[float] = (0.1, 0.5, 0.9)
    ) -> dict[str, Any]:
        """Generate predictions from a batch.

        Must return:
        ``{"mean": Tensor(B, H), "quantiles": {q: Tensor(B, H)}}``
        """
        ...

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _enrich_target(self, batch: dict) -> torch.Tensor:
        """Enrich past_target with covariate information (additive injection).

        Computes: ``enriched = target + gate * MLP(covariates)``

        The target signal is preserved exactly — covariates are added as a
        gated residual that starts at zero (gate initialized to 0) and grows
        as the model learns to use them. This avoids the target distortion
        problem of projection-based approaches.

        Returns
        -------
        torch.Tensor
            Enriched target of shape ``(B, L)``.
        """
        past = batch["past_target"]  # (B, L)
        if (
            self._cov_projector is not None
            and self._cov_gate is not None
            and "past_time_features" in batch
            and batch["past_time_features"].dim() >= 2
        ):
            ptf = batch["past_time_features"]  # (B, L, F) or (B, F)
            if ptf.dim() == 2 and ptf.size(0) == past.size(0):
                if ptf.size(1) != past.size(1):
                    ptf = ptf.unsqueeze(1).expand(-1, past.size(1), -1)
            if ptf.dim() == 3 and ptf.size(1) == past.size(1):
                # Additive: target + gate * MLP(covariates)
                cov_contribution = self._cov_projector(ptf).squeeze(-1)  # (B, L)
                past = past + torch.sigmoid(self._cov_gate) * cov_contribution
        return past

    def _resolve_device(self) -> torch.device:
        device = self.get_hyperparameter("device")
        if device is not None:
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _to_device(self, batch: dict) -> dict:
        """Move tensor values in a batch dict to the model's device."""
        out = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(self._device)
            else:
                out[k] = v
        return out

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader) -> float:
        """Compute average loss on a validation loader."""
        self._network.eval()
        total_loss = 0.0
        n = 0
        for batch in loader:
            batch = self._to_device(batch)
            loss = self._train_step(batch)
            total_loss += loss.item()
            n += 1
        self._network.train()
        return total_loss / max(n, 1)
