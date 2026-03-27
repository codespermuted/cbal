"""Step 2: TimeSeriesDataFrame verification tests."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cbal.dataset import TimeSeriesDataFrame


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_df():
    """Simple flat DataFrame with 2 items x 10 timestamps."""
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    return pd.DataFrame({
        "item_id": ["A"] * 10 + ["B"] * 10,
        "timestamp": list(dates) * 2,
        "target": list(range(10)) + list(range(10, 20)),
    })


@pytest.fixture
def sample_tsdf(sample_df):
    return TimeSeriesDataFrame.from_data_frame(sample_df)


# ---------------------------------------------------------------------------
# Factory: from_data_frame
# ---------------------------------------------------------------------------
class TestFromDataFrame:
    def test_basic_creation(self, sample_df):
        tsdf = TimeSeriesDataFrame.from_data_frame(sample_df)
        assert isinstance(tsdf, TimeSeriesDataFrame)
        assert isinstance(tsdf, pd.DataFrame)
        assert len(tsdf) == 20

    def test_custom_column_names(self):
        df = pd.DataFrame({
            "series_id": ["X"] * 5,
            "date": pd.date_range("2023-01-01", periods=5, freq="D"),
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        tsdf = TimeSeriesDataFrame.from_data_frame(
            df, id_column="series_id", timestamp_column="date", target_column="value"
        )
        assert tsdf.num_items == 1
        assert "target" in tsdf.columns

    def test_missing_column_raises(self, sample_df):
        with pytest.raises(ValueError, match="not found"):
            TimeSeriesDataFrame.from_data_frame(sample_df, id_column="nonexistent")

    def test_non_numeric_target_raises(self):
        df = pd.DataFrame({
            "item_id": ["A"] * 3,
            "timestamp": pd.date_range("2023-01-01", periods=3, freq="D"),
            "target": ["a", "b", "c"],
        })
        with pytest.raises(ValueError, match="numeric"):
            TimeSeriesDataFrame.from_data_frame(df)

    def test_duplicate_timestamps_raises(self):
        df = pd.DataFrame({
            "item_id": ["A", "A"],
            "timestamp": ["2023-01-01", "2023-01-01"],
            "target": [1.0, 2.0],
        })
        with pytest.raises(ValueError, match="duplicate"):
            TimeSeriesDataFrame.from_data_frame(df)

    def test_preserves_covariates(self):
        df = pd.DataFrame({
            "item_id": ["A"] * 5,
            "timestamp": pd.date_range("2023-01-01", periods=5, freq="D"),
            "target": [1.0, 2.0, 3.0, 4.0, 5.0],
            "covariate_1": [10, 20, 30, 40, 50],
        })
        tsdf = TimeSeriesDataFrame.from_data_frame(df)
        assert "covariate_1" in tsdf.columns

    def test_sorts_by_item_and_timestamp(self):
        df = pd.DataFrame({
            "item_id": ["B", "A", "B", "A"],
            "timestamp": ["2023-01-02", "2023-01-02", "2023-01-01", "2023-01-01"],
            "target": [4, 2, 3, 1],
        })
        tsdf = TimeSeriesDataFrame.from_data_frame(df)
        items = tsdf.index.get_level_values("item_id").tolist()
        assert items == ["A", "A", "B", "B"]


# ---------------------------------------------------------------------------
# Factory: from_path
# ---------------------------------------------------------------------------
class TestFromPath:
    def test_csv_loading(self, sample_df):
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            sample_df.to_csv(f.name, index=False)
            tsdf = TimeSeriesDataFrame.from_path(f.name)
        assert tsdf.num_items == 2
        assert len(tsdf) == 20


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------
class TestProperties:
    def test_item_ids(self, sample_tsdf):
        ids = sample_tsdf.item_ids
        assert set(ids) == {"A", "B"}

    def test_num_items(self, sample_tsdf):
        assert sample_tsdf.num_items == 2

    def test_freq_daily(self, sample_tsdf):
        assert sample_tsdf.freq == "D"

    def test_freq_hourly(self):
        df = pd.DataFrame({
            "item_id": ["A"] * 24,
            "timestamp": pd.date_range("2023-01-01", periods=24, freq="h"),
            "target": range(24),
        })
        tsdf = TimeSeriesDataFrame.from_data_frame(df)
        assert tsdf.freq == "h"

    def test_freq_monthly(self):
        df = pd.DataFrame({
            "item_id": ["A"] * 12,
            "timestamp": pd.date_range("2023-01-01", periods=12, freq="MS"),
            "target": range(12),
        })
        tsdf = TimeSeriesDataFrame.from_data_frame(df)
        assert tsdf.freq is not None  # Should infer MS or similar

    def test_num_timesteps_per_item(self, sample_tsdf):
        counts = sample_tsdf.num_timesteps_per_item
        assert counts["A"] == 10
        assert counts["B"] == 10


# ---------------------------------------------------------------------------
# train_test_split
# ---------------------------------------------------------------------------
class TestTrainTestSplit:
    def test_basic_split(self, sample_tsdf):
        train, test = sample_tsdf.train_test_split(prediction_length=3)
        # Train: 7 per item = 14 total
        assert len(train) == 14
        # Test: full data = 20 (AutoGluon convention)
        assert len(test) == 20

    def test_train_excludes_last_n(self, sample_tsdf):
        train, _ = sample_tsdf.train_test_split(prediction_length=3)
        # Last train timestamp for item A should be day 7 (0-indexed)
        item_a_train = train.loc["A"]
        assert len(item_a_train) == 7

    def test_split_preserves_type(self, sample_tsdf):
        train, test = sample_tsdf.train_test_split(prediction_length=3)
        assert isinstance(train, TimeSeriesDataFrame)
        assert isinstance(test, TimeSeriesDataFrame)

    def test_split_preserves_freq(self, sample_tsdf):
        train, test = sample_tsdf.train_test_split(prediction_length=3)
        assert train.freq == "D"
        assert test.freq == "D"

    def test_invalid_prediction_length(self, sample_tsdf):
        with pytest.raises(ValueError, match="prediction_length"):
            sample_tsdf.train_test_split(prediction_length=0)

    def test_prediction_length_too_large(self):
        """Items with <= prediction_length observations are skipped."""
        df = pd.DataFrame({
            "item_id": ["A"] * 3 + ["B"] * 10,
            "timestamp": list(pd.date_range("2023-01-01", periods=3, freq="D"))
                       + list(pd.date_range("2023-01-01", periods=10, freq="D")),
            "target": list(range(3)) + list(range(10)),
        })
        tsdf = TimeSeriesDataFrame.from_data_frame(df)
        train, test = tsdf.train_test_split(prediction_length=5)
        # Only item B should be in train (A has only 3 obs)
        assert set(train.item_ids) == {"B"}


# ---------------------------------------------------------------------------
# fill_missing_values
# ---------------------------------------------------------------------------
class TestFillMissing:
    def test_ffill(self):
        df = pd.DataFrame({
            "item_id": ["A"] * 5,
            "timestamp": pd.date_range("2023-01-01", periods=5, freq="D"),
            "target": [1.0, np.nan, np.nan, 4.0, 5.0],
        })
        tsdf = TimeSeriesDataFrame.from_data_frame(df)
        filled = tsdf.fill_missing_values(method="ffill")
        assert filled["target"].isna().sum() == 0
        assert filled["target"].iloc[1] == 1.0  # forward filled

    def test_zero(self):
        df = pd.DataFrame({
            "item_id": ["A"] * 3,
            "timestamp": pd.date_range("2023-01-01", periods=3, freq="D"),
            "target": [1.0, np.nan, 3.0],
        })
        tsdf = TimeSeriesDataFrame.from_data_frame(df)
        filled = tsdf.fill_missing_values(method="zero")
        assert filled["target"].iloc[1] == 0.0

    def test_unknown_method_raises(self, sample_tsdf):
        with pytest.raises(ValueError, match="Unknown fill method"):
            sample_tsdf.fill_missing_values(method="magic")


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
class TestSummary:
    def test_summary_shape(self, sample_tsdf):
        s = sample_tsdf.summary()
        assert len(s) == 2  # 2 items
        assert "count" in s.columns
        assert "mean" in s.columns
        assert "missing" in s.columns


# ---------------------------------------------------------------------------
# iter_items
# ---------------------------------------------------------------------------
class TestIterItems:
    def test_iter_items(self, sample_tsdf):
        items = list(sample_tsdf.iter_items())
        assert len(items) == 2
        assert items[0][0] == "A"
        assert items[1][0] == "B"


# ---------------------------------------------------------------------------
# repr
# ---------------------------------------------------------------------------
class TestRepr:
    def test_repr_contains_info(self, sample_tsdf):
        r = repr(sample_tsdf)
        assert "TimeSeriesDataFrame" in r
        assert "2 items" in r
        assert "20 rows" in r
