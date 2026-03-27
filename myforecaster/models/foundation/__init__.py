"""
Foundation Model Wrappers — Zero-shot pretrained time series models.

These models use HuggingFace pretrained weights and require NO training.
Each wraps a specific foundation model's API into our AbstractModel interface.

Supported models:
- **Chronos-2** (Amazon): pip install chronos-forecasting
- **TimesFM** (Google): pip install timesfm
- **Moirai** (Salesforce): pip install uni2ts
- **TTM** (IBM): pip install tsfm_public
- **Toto** (Databricks): pip install toto-benchmark

Usage::

    from myforecaster.models.foundation import Chronos2Model
    m = Chronos2Model(freq="D", prediction_length=24)
    m.fit(train_data)   # no-op (just stores metadata)
    pred = m.predict(train_data)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Sequence

import numpy as np
import pandas as pd

from myforecaster.dataset.ts_dataframe import ITEMID, TARGET, TIMESTAMP, TimeSeriesDataFrame
from myforecaster.models.abstract_model import AbstractTimeSeriesModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract Foundation Model
# ---------------------------------------------------------------------------

class AbstractFoundationModel(AbstractTimeSeriesModel):
    """Base class for zero-shot foundation model wrappers.

    Foundation models don't train — ``fit()`` just stores metadata.
    ``predict()`` calls the pretrained model's inference API.
    """

    _default_hyperparameters: dict[str, Any] = {
        "device": None,       # "cuda", "cpu", or None (auto)
        "batch_size": 32,
        "num_samples": 100,   # for probabilistic models
    }

    def _fit(self, train_data, val_data=None, time_limit=None):
        """No-op for foundation models (zero-shot)."""
        pass

    def _load_pipeline(self):
        """Lazy-load the model pipeline. Override in subclasses."""
        raise NotImplementedError

    def _get_device(self) -> str:
        device = self.get_hyperparameter("device")
        if device is not None:
            return device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"


# ============================================================================
# Chronos-2 (Amazon)
# ============================================================================

class Chronos2Model(AbstractFoundationModel):
    """Chronos-2: Universal forecasting foundation model (Amazon, 2025).

    120M encoder-only model with group attention for univariate, multivariate,
    and covariate-informed forecasting.

    Supports:
    - **Zero-shot**: No training, just inference (default)
    - **Covariates**: Pass known/past covariates as additional channels
    - **Fine-tuning**: Adapt pretrained weights to your data

    Requires: ``pip install chronos-forecasting``

    Other Parameters
    ----------------
    model_id : str
        HuggingFace model ID (default ``"amazon/chronos-2"``).
    finetune : bool
        If True, fine-tune on training data in ``fit()`` (default False).
    finetune_epochs : int
        Number of fine-tuning epochs (default 5).
    finetune_lr : float
        Fine-tuning learning rate (default 1e-4).
    use_covariates : bool
        If True, include known/past covariates as extra channels (default True).
    """

    _default_hyperparameters = {
        **AbstractFoundationModel._default_hyperparameters,
        "model_id": "amazon/chronos-2",
        "finetune": False,
        "finetune_epochs": 5,
        "finetune_lr": 1e-4,
        "finetune_batch_size": 16,
        "use_covariates": True,
    }

    def _load_pipeline(self):
        if hasattr(self, "_pipeline"):
            return self._pipeline
        try:
            from chronos import Chronos2Pipeline
        except ImportError:
            raise ImportError(
                "Chronos-2 requires: pip install chronos-forecasting"
            )
        model_path = getattr(self, "_finetuned_path", None) or self.get_hyperparameter("model_id")
        self._pipeline = Chronos2Pipeline.from_pretrained(
            model_path,
            device_map=self._get_device(),
        )
        return self._pipeline

    def _fit(self, train_data, val_data=None, time_limit=None):
        """Fine-tune Chronos-2 on training data (if finetune=True)."""
        if not self.get_hyperparameter("finetune"):
            # Zero-shot: store covariate names for predict
            self._known_cov_names = getattr(train_data, "known_covariates_names", []) or []
            self._past_cov_names = getattr(train_data, "past_covariates_names", []) or []
            return

        logger.info("Fine-tuning Chronos-2...")
        try:
            from chronos import Chronos2Pipeline
            from chronos.training import TrainingConfig, train as chronos_train
        except ImportError:
            try:
                # Alternative: manual fine-tuning via HuggingFace Trainer
                self._finetune_manual(train_data, val_data, time_limit)
                return
            except Exception as e:
                logger.warning(f"Chronos-2 fine-tuning not available: {e}. Using zero-shot.")
                return

        self._known_cov_names = getattr(train_data, "known_covariates_names", []) or []
        self._past_cov_names = getattr(train_data, "past_covariates_names", []) or []

        # Build training DataFrame with covariates
        train_df = self._build_chronos_df(train_data, include_covariates=True)

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                output_dir=tmpdir,
                num_train_epochs=self.get_hyperparameter("finetune_epochs"),
                learning_rate=self.get_hyperparameter("finetune_lr"),
                per_device_train_batch_size=self.get_hyperparameter("finetune_batch_size"),
                prediction_length=self.prediction_length,
                model_id=self.get_hyperparameter("model_id"),
            )
            chronos_train(config, train_df)
            self._finetuned_path = tmpdir
            # Reload pipeline with fine-tuned weights
            if hasattr(self, "_pipeline"):
                del self._pipeline
            self._load_pipeline()
            logger.info("Chronos-2 fine-tuning complete.")

    def _finetune_manual(self, train_data, val_data=None, time_limit=None):
        """Manual fine-tuning using HuggingFace Trainer when chronos.training
        is not available."""
        import torch
        from transformers import Trainer, TrainingArguments

        pipeline = self._load_pipeline()
        model = pipeline.model

        self._known_cov_names = getattr(train_data, "known_covariates_names", []) or []
        self._past_cov_names = getattr(train_data, "past_covariates_names", []) or []

        # Build dataset: list of context windows
        train_windows = self._build_training_windows(train_data)
        if not train_windows:
            logger.warning("No valid training windows for fine-tuning.")
            return

        epochs = self.get_hyperparameter("finetune_epochs")
        lr = self.get_hyperparameter("finetune_lr")
        batch_size = self.get_hyperparameter("finetune_batch_size")

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            training_args = TrainingArguments(
                output_dir=tmpdir,
                num_train_epochs=epochs,
                learning_rate=lr,
                per_device_train_batch_size=batch_size,
                save_strategy="no",
                logging_steps=10,
                remove_unused_columns=False,
                report_to="none",
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_windows,
            )
            trainer.train()

            # Save fine-tuned model
            model.save_pretrained(tmpdir)
            if hasattr(pipeline, "tokenizer"):
                pipeline.tokenizer.save_pretrained(tmpdir)
            self._finetuned_path = tmpdir

            # Reload pipeline
            if hasattr(self, "_pipeline"):
                del self._pipeline
            self._load_pipeline()

        logger.info(f"Chronos-2 manual fine-tuning complete ({epochs} epochs).")

    def _build_training_windows(self, train_data):
        """Build training windows from time series data for fine-tuning."""
        import torch
        from torch.utils.data import Dataset

        class ChronosFineTuneDataset(Dataset):
            def __init__(self, windows):
                self.windows = windows

            def __len__(self):
                return len(self.windows)

            def __getitem__(self, idx):
                return self.windows[idx]

        pred_len = self.prediction_length
        ctx_len = max(pred_len * 3, 64)
        total_len = ctx_len + pred_len

        windows = []
        for item_id in train_data.item_ids:
            item_df = train_data.loc[item_id]
            vals = item_df[TARGET].values.astype(np.float32)
            n = len(vals)

            # Slide windows
            stride = max(pred_len, 1)
            for start in range(0, n - total_len + 1, stride):
                window = vals[start : start + total_len]
                context = torch.tensor(window[:ctx_len])
                target = torch.tensor(window[ctx_len:])
                windows.append({
                    "input_ids": context,
                    "labels": target,
                })

        return ChronosFineTuneDataset(windows) if windows else None

    def _build_chronos_df(self, data, include_covariates=True):
        """Build DataFrame for Chronos-2 with optional covariate columns.

        Chronos-2 treats additional numeric columns as covariates
        (extra channels) automatically.
        """
        use_covs = include_covariates and self.get_hyperparameter("use_covariates")
        cov_cols = []
        if use_covs:
            cov_cols = list(getattr(self, "_known_cov_names", []) or [])
            cov_cols += list(getattr(self, "_past_cov_names", []) or [])
            actual_cols = set(data.columns) - {TARGET}
            cov_cols = [c for c in cov_cols if c in actual_cols]

        frames = []
        for item_id in data.item_ids:
            item_df = data.loc[item_id]
            ts = item_df.index.get_level_values(TIMESTAMP)
            n = len(ts)

            # Infer frequency from timestamps
            if n >= 2:
                freq = pd.infer_freq(ts)
                if freq is None:
                    # Fallback: compute median diff
                    diffs = pd.Series(ts).diff().dropna()
                    median_diff = diffs.median()
                    if median_diff <= pd.Timedelta(minutes=15):
                        freq = "15min"
                    elif median_diff <= pd.Timedelta(hours=1):
                        freq = "h"
                    elif median_diff <= pd.Timedelta(days=1):
                        freq = "D"
                    else:
                        freq = "h"
                # Rebuild timestamps with proper frequency
                ts_freq = pd.date_range(start=ts[0], periods=n, freq=freq)
            else:
                ts_freq = ts

            item_frame = pd.DataFrame({
                "id": item_id,
                "timestamp": ts_freq,
                "target": item_df[TARGET].values.astype(float),
            })
            for col in cov_cols:
                item_frame[col] = item_df[col].values.astype(float)
            frames.append(item_frame)

        return pd.concat(frames, ignore_index=True)

    def _predict(self, data, known_covariates=None, quantile_levels=None):
        quantile_levels = quantile_levels or [0.1, 0.5, 0.9]
        pipeline = self._load_pipeline()

        use_covs = self.get_hyperparameter("use_covariates")
        known_cov_names = getattr(self, "_known_cov_names", []) or []
        past_cov_names = getattr(self, "_past_cov_names", []) or []

        # Build context DataFrame with covariates
        context_df = self._build_chronos_df(data, include_covariates=use_covs)

        # Build known future covariates DataFrame (if available)
        # Chronos-2 can accept future covariate values for the prediction horizon
        future_cov_df = None
        if use_covs and known_cov_names and known_covariates is not None:
            fcov_rows = []
            actual_cols = set(known_covariates.columns)
            valid_cov_cols = [c for c in known_cov_names if c in actual_cols]
            if valid_cov_cols:
                for item_id in data.item_ids:
                    try:
                        item_cov = known_covariates.loc[item_id]
                        cov_ts = item_cov.index.get_level_values(TIMESTAMP)
                        for i, t in enumerate(cov_ts):
                            row = {"id": item_id, "timestamp": t}
                            for col in valid_cov_cols:
                                row[col] = float(item_cov[col].values[i])
                            fcov_rows.append(row)
                    except (KeyError, IndexError):
                        pass
                if fcov_rows:
                    future_cov_df = pd.DataFrame(fcov_rows)

        # Determine covariate column names for Chronos-2
        cov_col_list = []
        if use_covs:
            actual_context_cols = set(context_df.columns) - {"id", "timestamp", "target"}
            cov_col_list = sorted(actual_context_cols)

        # Build predict_df kwargs
        predict_kwargs = {
            "prediction_length": self.prediction_length,
            "quantile_levels": quantile_levels,
            "id_column": "id",
            "timestamp_column": "timestamp",
            "target": "target",
        }

        # Pass covariates if available
        if cov_col_list:
            predict_kwargs["covariates"] = cov_col_list

        if future_cov_df is not None:
            predict_kwargs["future_covariates"] = future_cov_df

        try:
            pred_df = pipeline.predict_df(context_df, **predict_kwargs)
        except TypeError:
            # Fallback: older chronos version without covariate support
            pred_df = pipeline.predict_df(
                context_df,
                prediction_length=self.prediction_length,
                quantile_levels=quantile_levels,
                id_column="id",
                timestamp_column="timestamp",
                target="target",
            )

        return self._chronos_to_tsdf(pred_df, data, quantile_levels)

    def _chronos_to_tsdf(self, pred_df, data, quantile_levels):
        """Convert Chronos-2 output DataFrame to our TimeSeriesDataFrame."""
        all_rows = []
        for item_id in data.item_ids:
            item_pred = pred_df[pred_df["id"] == item_id]
            future_ts = self._make_future_timestamps(data, item_id)
            for h in range(min(self.prediction_length, len(item_pred))):
                row = {
                    ITEMID: item_id,
                    TIMESTAMP: future_ts[h],
                    "mean": item_pred.iloc[h].get("0.5", item_pred.iloc[h].get("mean", 0)),
                }
                for q in quantile_levels:
                    col = str(q)
                    row[col] = item_pred.iloc[h].get(col, row["mean"])
                all_rows.append(row)
        return self._rows_to_tsdf(all_rows)


# ============================================================================
# TimesFM (Google)
# ============================================================================

class TimesFMModel(AbstractFoundationModel):
    """TimesFM: Decoder-only time series foundation model (Google, ICML 2024).

    200M parameter model with patching and quantile head.

    Requires: ``pip install timesfm``

    Other Parameters
    ----------------
    model_id : str
        HuggingFace model ID (default ``"google/timesfm-2.5-200m-pytorch"``).
    """

    _default_hyperparameters = {
        **AbstractFoundationModel._default_hyperparameters,
        "model_id": "google/timesfm-2.5-200m-pytorch",
    }

    def _load_pipeline(self):
        if hasattr(self, "_model"):
            return self._model
        try:
            import timesfm
        except ImportError:
            raise ImportError("TimesFM requires: pip install timesfm")

        self._model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            self.get_hyperparameter("model_id"),
        )
        self._model.compile(
            timesfm.ForecastConfig(
                max_context=1024,
                max_horizon=max(256, self.prediction_length),
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
            )
        )
        return self._model

    def _predict(self, data, known_covariates=None, quantile_levels=None):
        quantile_levels = quantile_levels or [0.1, 0.5, 0.9]
        model = self._load_pipeline()

        all_rows = []
        inputs = []
        item_ids = list(data.item_ids)

        for item_id in item_ids:
            item_df = data.loc[item_id]
            vals = item_df[TARGET].values.astype(np.float64)
            inputs.append(vals)

        point_forecast, quantile_forecast = model.forecast(
            horizon=self.prediction_length,
            inputs=inputs,
        )
        # point_forecast: (N, H), quantile_forecast: (N, H, 10) [mean + 9 quantiles]

        # TimesFM quantiles: index 0=mean, 1-9 = 10th,20th,...,90th percentiles
        tfm_quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        for i, item_id in enumerate(item_ids):
            future_ts = self._make_future_timestamps(data, item_id)
            for h in range(self.prediction_length):
                row = {
                    ITEMID: item_id,
                    TIMESTAMP: future_ts[h],
                    "mean": float(point_forecast[i, h]),
                }
                for q in quantile_levels:
                    # Find closest TimesFM quantile index
                    q_idx = min(range(len(tfm_quantiles)),
                                key=lambda j: abs(tfm_quantiles[j] - q))
                    row[str(q)] = float(quantile_forecast[i, h, q_idx + 1])
                all_rows.append(row)

        return self._rows_to_tsdf(all_rows)


# ============================================================================
# Moirai (Salesforce)
# ============================================================================

class MoiraiModel(AbstractFoundationModel):
    """Moirai: Universal time series forecasting transformer (Salesforce, ICML 2024).

    Any-variate model with mixture distribution output.

    Requires: ``pip install uni2ts``

    Other Parameters
    ----------------
    model_size : str
        ``"small"``, ``"base"``, or ``"large"`` (default ``"small"``).
    model_version : str
        ``"moirai2"``, ``"moirai"`` (default ``"moirai2"``).
    """

    _default_hyperparameters = {
        **AbstractFoundationModel._default_hyperparameters,
        "model_size": "small",
        "model_version": "moirai2",
        "context_length": 200,
        "patch_size": "auto",
        "num_samples": 100,
    }

    def _load_pipeline(self):
        if hasattr(self, "_forecast_model"):
            return self._forecast_model
        try:
            from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
        except ImportError:
            raise ImportError("Moirai requires: pip install uni2ts")

        version = self.get_hyperparameter("model_version")
        size = self.get_hyperparameter("model_size")

        if version == "moirai2":
            repo = f"Salesforce/moirai-2-R-{size}"
        else:
            repo = f"Salesforce/moirai-1.0-R-{size}"

        from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
        module = MoiraiModule.from_pretrained(repo)
        self._forecast_model = MoiraiForecast(
            module=module,
            prediction_length=self.prediction_length,
            context_length=self.get_hyperparameter("context_length"),
            patch_size=self.get_hyperparameter("patch_size"),
            num_samples=self.get_hyperparameter("num_samples"),
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        return self._forecast_model

    def _predict(self, data, known_covariates=None, quantile_levels=None):
        quantile_levels = quantile_levels or [0.1, 0.5, 0.9]
        forecast_model = self._load_pipeline()

        try:
            from gluonts.dataset.pandas import PandasDataset
            from gluonts.dataset.split import split
        except ImportError:
            raise ImportError("Moirai requires gluonts: pip install gluonts")

        # Convert to GluonTS PandasDataset
        frames = {}
        for item_id in data.item_ids:
            item_df = data.loc[item_id]
            ts = item_df.index.get_level_values(TIMESTAMP)
            vals = item_df[TARGET].values.astype(np.float64)
            frames[item_id] = pd.DataFrame({"target": vals}, index=ts)

        wide_df = pd.concat(frames, axis=1)
        # Use long format for GluonTS
        long_rows = []
        for item_id in data.item_ids:
            item_df = data.loc[item_id]
            ts = item_df.index.get_level_values(TIMESTAMP)
            vals = item_df[TARGET].values
            for t, v in zip(ts, vals):
                long_rows.append({"item_id": item_id, "timestamp": t, "target": float(v)})
        long_df = pd.DataFrame(long_rows)
        long_df = long_df.set_index("timestamp")

        ds = PandasDataset.from_long_dataframe(long_df, target="target", item_id="item_id")

        # Create test instances (predict from end of each series)
        _, test_template = split(ds, offset=-self.prediction_length)
        test_data = test_template.generate_instances(
            prediction_length=self.prediction_length, windows=1
        )

        # Run inference
        predictor = forecast_model.create_predictor(batch_size=self.get_hyperparameter("batch_size"))
        forecasts = list(predictor.predict(test_data.input))

        # Convert forecasts to our format
        all_rows = []
        for i, (item_id, forecast) in enumerate(zip(data.item_ids, forecasts)):
            future_ts = self._make_future_timestamps(data, item_id)
            samples = forecast.samples  # (num_samples, H)
            mean = samples.mean(axis=0)
            for h in range(self.prediction_length):
                row = {
                    ITEMID: item_id,
                    TIMESTAMP: future_ts[h],
                    "mean": float(mean[h]),
                }
                for q in quantile_levels:
                    row[str(q)] = float(np.quantile(samples[:, h], q))
                all_rows.append(row)

        return self._rows_to_tsdf(all_rows)


# ============================================================================
# TTM — Tiny Time Mixers (IBM)
# ============================================================================

class TTMModel(AbstractFoundationModel):
    """TTM: Tiny Time Mixers — lightweight foundation model (IBM, 2024).

    ~1M parameters, CPU-friendly. Based on TSMixer architecture.

    Requires: ``pip install tsfm_public``

    Other Parameters
    ----------------
    model_id : str
        HuggingFace model ID (default ``"ibm-granite/granite-timeseries-ttm-r2"``).
    context_length : int
        512, 1024, or 1536 (default 512).
    """

    _default_hyperparameters = {
        **AbstractFoundationModel._default_hyperparameters,
        "model_id": "ibm-granite/granite-timeseries-ttm-r2",
        "context_length": 512,
    }

    def _load_pipeline(self):
        if hasattr(self, "_model"):
            return self._model
        try:
            from tsfm_public import TinyTimeMixerForPrediction
        except ImportError:
            raise ImportError("TTM requires: pip install tsfm_public")

        ctx = self.get_hyperparameter("context_length")
        model_id = self.get_hyperparameter("model_id")

        self._model = TinyTimeMixerForPrediction.from_pretrained(
            model_id,
            prediction_filter_length=self.prediction_length,
            revision=f"main-r2-{ctx}",
        )
        self._model.eval()
        return self._model

    def _predict(self, data, known_covariates=None, quantile_levels=None):
        quantile_levels = quantile_levels or [0.1, 0.5, 0.9]
        model = self._load_pipeline()

        try:
            import torch
        except ImportError:
            raise ImportError("TTM requires PyTorch")

        ctx_len = self.get_hyperparameter("context_length")
        all_rows = []

        for item_id in data.item_ids:
            item_df = data.loc[item_id]
            vals = item_df[TARGET].values.astype(np.float64)

            # Pad/trim to context_length
            if len(vals) >= ctx_len:
                context = vals[-ctx_len:]
            else:
                context = np.pad(vals, (ctx_len - len(vals), 0), mode="edge")

            # TTM expects (batch, context_length, n_channels)
            x = torch.tensor(context, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

            with torch.no_grad():
                output = model(x)

            # output.prediction_outputs: (1, pred_len, 1)
            pred = output.prediction_outputs.squeeze().cpu().numpy()
            if pred.ndim == 2:
                pred = pred[:, 0]  # take first channel
            pred = pred[:self.prediction_length]

            future_ts = self._make_future_timestamps(data, item_id)
            for h in range(self.prediction_length):
                row = {
                    ITEMID: item_id,
                    TIMESTAMP: future_ts[h],
                    "mean": float(pred[h]),
                }
                for q in quantile_levels:
                    row[str(q)] = float(pred[h])  # deterministic model
                all_rows.append(row)

        return self._rows_to_tsdf(all_rows)


# ============================================================================
# Toto (Databricks / Datadog)
# ============================================================================

class TotoModel(AbstractFoundationModel):
    """Toto: Time series foundation model (Databricks, 2024).

    Decoder-only transformer with mixture output.

    Requires: ``pip install toto-benchmark`` or custom loading.

    Other Parameters
    ----------------
    model_id : str
        HuggingFace model ID (default ``"Datadog/toto-open-base-1.0"``).
    """

    _default_hyperparameters = {
        **AbstractFoundationModel._default_hyperparameters,
        "model_id": "Datadog/toto-open-base-1.0",
        "context_length": 1024,
    }

    def _load_pipeline(self):
        if hasattr(self, "_model"):
            return self._model
        try:
            import torch
            from huggingface_hub import hf_hub_download
            # Toto uses a custom model class; attempt to load via torch
            model_id = self.get_hyperparameter("model_id")
            # Try toto-specific import first
            try:
                from toto import TotoForForecasting
                self._model = TotoForForecasting.from_pretrained(model_id)
            except ImportError:
                # Fallback: load via AutoModel
                from transformers import AutoModelForCausalLM
                self._model = AutoModelForCausalLM.from_pretrained(
                    model_id, trust_remote_code=True
                )
            self._model.eval()
            device = self._get_device()
            if device == "cuda":
                self._model = self._model.cuda()
        except Exception as e:
            raise ImportError(
                f"Toto model loading failed: {e}. "
                "Try: pip install toto-benchmark or check model access."
            )
        return self._model

    def _predict(self, data, known_covariates=None, quantile_levels=None):
        quantile_levels = quantile_levels or [0.1, 0.5, 0.9]
        model = self._load_pipeline()
        import torch

        ctx_len = self.get_hyperparameter("context_length")
        all_rows = []

        for item_id in data.item_ids:
            item_df = data.loc[item_id]
            vals = item_df[TARGET].values.astype(np.float64)

            if len(vals) >= ctx_len:
                context = vals[-ctx_len:]
            else:
                context = np.pad(vals, (ctx_len - len(vals), 0), mode="edge")

            x = torch.tensor(context, dtype=torch.float32).unsqueeze(0)
            device = next(model.parameters()).device
            x = x.to(device)

            with torch.no_grad():
                output = model.generate(x, prediction_length=self.prediction_length)

            if hasattr(output, "samples"):
                samples = output.samples.cpu().numpy()  # (n_samples, H)
                pred_mean = samples.mean(axis=0)
            elif hasattr(output, "sequences"):
                pred_mean = output.sequences[0].cpu().numpy()[-self.prediction_length:]
                samples = None
            else:
                pred_mean = output.cpu().numpy().flatten()[-self.prediction_length:]
                samples = None

            future_ts = self._make_future_timestamps(data, item_id)
            for h in range(self.prediction_length):
                row = {
                    ITEMID: item_id,
                    TIMESTAMP: future_ts[h],
                    "mean": float(pred_mean[h]) if h < len(pred_mean) else 0.0,
                }
                for q in quantile_levels:
                    if samples is not None:
                        row[str(q)] = float(np.quantile(samples[:, h], q))
                    else:
                        row[str(q)] = row["mean"]
                all_rows.append(row)

        return self._rows_to_tsdf(all_rows)


# ============================================================================
# Registration
# ============================================================================

from myforecaster.models import register_model

register_model("Chronos-2")(Chronos2Model)
register_model("TimesFM")(TimesFMModel)
register_model("Moirai")(MoiraiModel)
register_model("TTM")(TTMModel)
register_model("Toto")(TotoModel)
