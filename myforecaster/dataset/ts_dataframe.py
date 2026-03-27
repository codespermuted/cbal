"""
TimeSeriesDataFrame — Multi-series time series data container.

Inspired by AutoGluon's TimeSeriesDataFrame, this provides a pandas-based
container with a (item_id, timestamp) MultiIndex and time-series-specific
utilities like frequency inference, train/test splitting, and validation.

Usage::

    from myforecaster.dataset import TimeSeriesDataFrame

    tsdf = TimeSeriesDataFrame.from_data_frame(
        df, id_column="item_id", timestamp_column="timestamp", target_column="target"
    )
    print(tsdf.num_items, tsdf.freq)
    train, test = tsdf.train_test_split(prediction_length=7)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ITEMID = "item_id"
TIMESTAMP = "timestamp"
TARGET = "target"


class TimeSeriesDataFrame(pd.DataFrame):
    """A DataFrame with a ``(item_id, timestamp)`` MultiIndex.

    Each row represents one observation for one time series at one point in
    time.  The required column is ``target`` (the value to forecast).
    Additional columns are treated as covariates.

    Parameters
    ----------
    *args, **kwargs
        Forwarded to ``pd.DataFrame.__init__``.  The resulting frame **must**
        have a two-level MultiIndex named ``(item_id, timestamp)``.
    """

    # ------------------------------------------------------------------
    # pandas subclass boilerplate
    # ------------------------------------------------------------------
    _metadata = ["_cached_freq", "_static_features",
                 "_known_covariates_names", "_past_covariates_names"]

    @property
    def _constructor(self):
        return TimeSeriesDataFrame

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not hasattr(self, "_cached_freq"):
            self._cached_freq: str | None = None
        if not hasattr(self, "_static_features"):
            self._static_features: pd.DataFrame | None = None
        if not hasattr(self, "_known_covariates_names"):
            self._known_covariates_names: list[str] | None = None
        if not hasattr(self, "_past_covariates_names"):
            self._past_covariates_names: list[str] | None = None

    # ------------------------------------------------------------------
    # Static features (persisted through copy/slice)
    # ------------------------------------------------------------------
    @property
    def static_features(self) -> pd.DataFrame | None:
        """Per-item static metadata (time-independent features).

        A DataFrame indexed by item_id with one row per item.
        Automatically filtered when the TSDF is sliced.
        """
        return self._static_features

    @static_features.setter
    def static_features(self, value: pd.DataFrame | None):
        if value is not None:
            if not isinstance(value, pd.DataFrame):
                raise TypeError("static_features must be a DataFrame or None")
            value = value.copy()
            if value.index.name != ITEMID:
                value.index.name = ITEMID
            value.index = value.index.astype(str)
            # Validate: every item in TSDF must have a static row
            missing = set(self.item_ids) - set(value.index)
            if missing:
                logger.warning(
                    f"static_features missing for items: "
                    f"{list(missing)[:5]}{'...' if len(missing) > 5 else ''}"
                )
        self._static_features = value

    # ------------------------------------------------------------------
    # Covariate column tracking
    # ------------------------------------------------------------------
    @property
    def known_covariates_names(self) -> list[str]:
        """Column names of known-future covariates."""
        return list(self._known_covariates_names or [])

    @known_covariates_names.setter
    def known_covariates_names(self, value: list[str] | None):
        if value:
            missing = [c for c in value if c not in self.columns]
            if missing:
                raise ValueError(
                    f"known_covariates_names {missing} not in columns {list(self.columns)}"
                )
        self._known_covariates_names = list(value) if value else None

    @property
    def past_covariates_names(self) -> list[str]:
        """Column names of past-only covariates."""
        return list(self._past_covariates_names or [])

    @past_covariates_names.setter
    def past_covariates_names(self, value: list[str] | None):
        if value:
            missing = [c for c in value if c not in self.columns]
            if missing:
                raise ValueError(
                    f"past_covariates_names {missing} not in columns {list(self.columns)}"
                )
        self._past_covariates_names = list(value) if value else None

    @property
    def covariate_columns(self) -> list[str]:
        """All non-target data columns (known + past + untracked)."""
        return [c for c in self.columns if c != TARGET]

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------
    @classmethod
    def from_data_frame(
        cls,
        df: pd.DataFrame,
        id_column: str = ITEMID,
        timestamp_column: str = TIMESTAMP,
        target_column: str = TARGET,
        *,
        static_features: pd.DataFrame | None = None,
    ) -> "TimeSeriesDataFrame":
        """Create a ``TimeSeriesDataFrame`` from a flat DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Input data with at least ``id_column``, ``timestamp_column``, and
            ``target_column``.
        id_column : str
            Column identifying each individual time series.
        timestamp_column : str
            Column containing timestamps (will be converted to ``datetime``).
        target_column : str
            Column containing the values to forecast.
        static_features : pd.DataFrame, optional
            Per-item static features indexed by item_id.

        Returns
        -------
        TimeSeriesDataFrame
        """
        df = df.copy()

        # Validate required columns
        for col, label in [
            (id_column, "id_column"),
            (timestamp_column, "timestamp_column"),
            (target_column, "target_column"),
        ]:
            if col not in df.columns:
                raise ValueError(f"{label}={col!r} not found in DataFrame columns: {list(df.columns)}")

        # Rename to canonical names
        rename_map = {}
        if id_column != ITEMID:
            rename_map[id_column] = ITEMID
        if timestamp_column != TIMESTAMP:
            rename_map[timestamp_column] = TIMESTAMP
        if target_column != TARGET:
            rename_map[target_column] = TARGET
        if rename_map:
            df = df.rename(columns=rename_map)

        # Ensure types
        df[TIMESTAMP] = pd.to_datetime(df[TIMESTAMP])
        df[ITEMID] = df[ITEMID].astype(str)

        # Sort and set index
        df = df.sort_values([ITEMID, TIMESTAMP]).reset_index(drop=True)
        df = df.set_index([ITEMID, TIMESTAMP])

        tsdf = cls(df)
        tsdf._validate_structure()

        if static_features is not None:
            tsdf.static_features = static_features

        return tsdf

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        id_column: str = ITEMID,
        timestamp_column: str = TIMESTAMP,
        target_column: str = TARGET,
        **read_csv_kwargs,
    ) -> "TimeSeriesDataFrame":
        """Create a ``TimeSeriesDataFrame`` from a CSV file.

        Parameters
        ----------
        path : str or Path
            Path to CSV file (local path or URL).
        id_column, timestamp_column, target_column : str
            Column names (same as :meth:`from_data_frame`).
        **read_csv_kwargs
            Extra keyword arguments forwarded to ``pd.read_csv``.

        Returns
        -------
        TimeSeriesDataFrame
        """
        df = pd.read_csv(str(path), **read_csv_kwargs)
        return cls.from_data_frame(df, id_column, timestamp_column, target_column)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def item_ids(self) -> np.ndarray:
        """Unique item identifiers."""
        return self.index.get_level_values(ITEMID).unique().values

    @property
    def num_items(self) -> int:
        """Number of distinct time series."""
        return len(self.item_ids)

    @property
    def freq(self) -> str | None:
        """Inferred frequency of the time series (e.g. 'D', 'h', 'MS').

        Inferred from the first item; returns ``None`` when inference fails.
        """
        if self._cached_freq is not None:
            return self._cached_freq
        self._cached_freq = self._infer_freq()
        return self._cached_freq

    def _propagate_metadata(self, other: "TimeSeriesDataFrame"):
        """Copy metadata from self to other."""
        other._cached_freq = self.freq
        other._static_features = self._static_features
        other._known_covariates_names = self._known_covariates_names
        other._past_covariates_names = self._past_covariates_names

    @property
    def num_timesteps_per_item(self) -> pd.Series:
        """Number of observations per item."""
        return self.groupby(level=ITEMID).size().rename("num_timesteps")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def iter_items(self):
        """Iterate over ``(item_id, sub_dataframe)`` pairs."""
        for item_id in self.item_ids:
            yield item_id, self.loc[item_id]

    def slice_by_timestep(
        self, start: int | None = None, end: int | None = None,
    ) -> "TimeSeriesDataFrame":
        """Slice each item by integer position along the time axis.

        Parameters
        ----------
        start, end : int or None
            Python-style slice indices (negative allowed).
        """
        slices = []
        for item_id in self.item_ids:
            item_df = self.loc[[item_id]]
            slices.append(item_df.iloc[start:end])
        result = TimeSeriesDataFrame(pd.concat(slices))
        self._propagate_metadata(result)
        return result

    def get_model_inputs_for_scoring(
        self,
        prediction_length: int,
        known_covariates_names: list[str] | None = None,
    ) -> tuple["TimeSeriesDataFrame", "TimeSeriesDataFrame | None"]:
        """Prepare model inputs to predict the last ``prediction_length`` steps.

        Returns
        -------
        past_data : TimeSeriesDataFrame
            Data with last ``prediction_length`` steps removed.
        known_covariates : TimeSeriesDataFrame or None
            Future covariate values (if names provided).
        """
        past_data = self.slice_by_timestep(None, -prediction_length)
        kc_names = known_covariates_names or self.known_covariates_names
        if kc_names:
            future = self.slice_by_timestep(-prediction_length, None)
            kc = future[kc_names].copy()
            known_covariates = TimeSeriesDataFrame(kc)
            known_covariates._cached_freq = self.freq
        else:
            known_covariates = None
        return past_data, known_covariates

    def train_test_split(
        self, prediction_length: int
    ) -> tuple["TimeSeriesDataFrame", "TimeSeriesDataFrame"]:
        """Split each item so the last ``prediction_length`` steps go to test.

        Parameters
        ----------
        prediction_length : int
            Number of time steps to hold out per item.

        Returns
        -------
        train : TimeSeriesDataFrame
            All observations except the last ``prediction_length`` per item.
        test : TimeSeriesDataFrame
            The **full** series (context + future) — same convention as
            AutoGluon where the test set contains the complete history so
            that models can compute features.
        """
        if prediction_length < 1:
            raise ValueError(f"prediction_length must be >= 1, got {prediction_length}")

        train_slices = []
        for item_id in self.item_ids:
            item_df = self.loc[[item_id]]  # keep MultiIndex
            n = len(item_df)
            if n <= prediction_length:
                logger.warning(
                    f"Item {item_id!r} has only {n} observations "
                    f"(<= prediction_length={prediction_length}). Skipping from train."
                )
                continue
            train_slices.append(item_df.iloc[: n - prediction_length])

        if not train_slices:
            raise ValueError("No items have enough observations for the given prediction_length.")

        train = TimeSeriesDataFrame(pd.concat(train_slices))
        self._propagate_metadata(train)

        # test = full data (same AutoGluon convention)
        test = self.copy()
        self._propagate_metadata(test)

        return train, test

    def multi_window_backtest_splits(
        self, prediction_length: int, num_windows: int = 1,
        val_step_size: int | None = None,
    ) -> list[tuple["TimeSeriesDataFrame", "TimeSeriesDataFrame"]]:
        """Generate multiple (train, test) splits for backtesting.

        Creates ``num_windows`` splits by sliding the cutoff backward in
        increments of ``val_step_size`` (defaults to ``prediction_length``).

        Parameters
        ----------
        prediction_length : int
        num_windows : int (default 1)
        val_step_size : int, optional
            Step between successive validation windows.  Default = prediction_length.

        Returns
        -------
        list of (train_tsdf, test_tsdf)
            Oldest window first, newest window last.
        """
        step = val_step_size or prediction_length
        splits = []
        for w in range(num_windows - 1, -1, -1):
            offset = step * w
            train_slices, test_slices = [], []
            for item_id in self.item_ids:
                item_df = self.loc[[item_id]]
                n = len(item_df)
                test_end = n - offset if w > 0 else n
                train_end = test_end - prediction_length
                if train_end <= 0:
                    continue
                test_slices.append(item_df.iloc[:test_end])
                train_slices.append(item_df.iloc[:train_end])
            if not train_slices:
                continue
            train = TimeSeriesDataFrame(pd.concat(train_slices))
            self._propagate_metadata(train)
            test = TimeSeriesDataFrame(pd.concat(test_slices))
            self._propagate_metadata(test)
            splits.append((train, test))
        return splits

    def fill_missing_values(self, method: str = "ffill") -> "TimeSeriesDataFrame":
        """Fill missing target values.

        Parameters
        ----------
        method : str
            One of ``'ffill'``, ``'bfill'``, ``'zero'``, ``'mean'``.

        Returns
        -------
        TimeSeriesDataFrame
        """
        df = self.copy()
        if method == "ffill":
            df[TARGET] = df.groupby(level=ITEMID)[TARGET].ffill()
        elif method == "bfill":
            df[TARGET] = df.groupby(level=ITEMID)[TARGET].bfill()
        elif method == "zero":
            df[TARGET] = df[TARGET].fillna(0.0)
        elif method == "mean":
            means = df.groupby(level=ITEMID)[TARGET].transform("mean")
            df[TARGET] = df[TARGET].fillna(means)
        else:
            raise ValueError(f"Unknown fill method: {method!r}. Use 'ffill', 'bfill', 'zero', or 'mean'.")
        return TimeSeriesDataFrame(df)

    def summary(self) -> pd.DataFrame:
        """Return per-item summary statistics."""
        grouped = self[TARGET].groupby(level=ITEMID)
        return pd.DataFrame({
            "count": grouped.count(),
            "mean": grouped.mean(),
            "std": grouped.std(),
            "min": grouped.min(),
            "max": grouped.max(),
            "missing": grouped.apply(lambda s: s.isna().sum()),
        })

    def convert_frequency(
        self,
        freq: str,
        agg_numeric: str = "mean",
        agg_categorical: str = "last",
    ) -> "TimeSeriesDataFrame":
        """Resample all time series to a new frequency.

        Upsampling (e.g. monthly→daily) fills with NaN.
        Downsampling (e.g. daily→weekly) aggregates.

        Parameters
        ----------
        freq : str
            Target frequency (e.g. ``"W"``, ``"MS"``, ``"D"``).
        agg_numeric : str
            Aggregation for numeric columns (default ``"mean"``).
        agg_categorical : str
            Aggregation for categorical/object columns (default ``"last"``).
        """
        resampled_parts = []
        for item_id in self.item_ids:
            item_df = self.loc[item_id].copy()
            # Separate numeric and categorical
            num_cols = item_df.select_dtypes(include="number").columns.tolist()
            cat_cols = item_df.select_dtypes(include=["object", "category"]).columns.tolist()
            agg_dict = {c: agg_numeric for c in num_cols}
            agg_dict.update({c: agg_categorical for c in cat_cols})
            if not agg_dict:
                continue
            resampled = item_df.resample(freq).agg(agg_dict)
            resampled[ITEMID] = item_id
            resampled.index.name = TIMESTAMP
            resampled = resampled.reset_index().set_index([ITEMID, TIMESTAMP])
            resampled_parts.append(resampled)

        if not resampled_parts:
            raise ValueError("No items to resample.")
        result = TimeSeriesDataFrame(pd.concat(resampled_parts))
        result._cached_freq = freq
        self._propagate_metadata(result)
        result._cached_freq = freq  # override with new freq
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _validate_structure(self):
        """Check that the DataFrame has the expected structure."""
        # Must have MultiIndex with correct names
        if not isinstance(self.index, pd.MultiIndex):
            raise ValueError("TimeSeriesDataFrame must have a MultiIndex (item_id, timestamp).")
        if list(self.index.names) != [ITEMID, TIMESTAMP]:
            raise ValueError(
                f"MultiIndex levels must be named {[ITEMID, TIMESTAMP]}, "
                f"got {list(self.index.names)}"
            )

        # Must have 'target' column
        if TARGET not in self.columns:
            raise ValueError(f"Column '{TARGET}' is required but not found. Columns: {list(self.columns)}")

        # Target must be numeric
        if not pd.api.types.is_numeric_dtype(self[TARGET]):
            raise ValueError(f"Column '{TARGET}' must be numeric, got dtype {self[TARGET].dtype}")

        # Timestamps must be datetime
        ts_level = self.index.get_level_values(TIMESTAMP)
        if not pd.api.types.is_datetime64_any_dtype(ts_level):
            raise ValueError(f"Timestamp index level must be datetime, got {ts_level.dtype}")

        # Check for duplicate (item_id, timestamp) pairs
        if self.index.duplicated().any():
            n_dup = self.index.duplicated().sum()
            raise ValueError(f"Found {n_dup} duplicate (item_id, timestamp) pairs.")

    def _infer_freq(self) -> str | None:
        """Infer time series frequency from the first item."""
        if self.empty:
            return None
        first_item = self.item_ids[0]
        timestamps = self.loc[first_item].index.get_level_values(TIMESTAMP) if isinstance(
            self.index, pd.MultiIndex
        ) else self.loc[first_item].index
        if len(timestamps) < 3:
            return None
        try:
            freq = pd.infer_freq(timestamps)
            return freq
        except (ValueError, TypeError):
            return None

    def __repr__(self) -> str:
        header = (
            f"TimeSeriesDataFrame with {self.num_items} items, "
            f"{len(self)} rows, freq={self.freq!r}"
        )
        return f"{header}\n{super().__repr__()}"
