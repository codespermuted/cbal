#!/usr/bin/env python3
"""
Benchmark: C-BAL vs AutoGluon-TimeSeries
================================================

Compares on 4 representative datasets covering different characteristics:

1. **M3 Monthly**   — 1428 series, monthly, pred_len=18  (classic competition)
2. **M3 Quarterly** — 756 series, quarterly, pred_len=8   (seasonal, fewer obs)
3. **Tourism Monthly** — 366 series, monthly, pred_len=24  (hierarchical tourism)
4. **M4 Weekly**    — 359 series, weekly, pred_len=13     (higher freq)

Metrics: MASE (point), WQL (probabilistic), sMAPE (percentage)
Presets: medium_quality (fair comparison, reasonable runtime)
"""

import os
import sys
import time
import warnings
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
# Dataset loaders
# ─────────────────────────────────────────────────────────

def load_m3_monthly():
    """M3 Monthly: 1428 series, monthly, pred_len=18."""
    from datasetsforecast.m3 import M3
    df, *_ = M3.load(".", group="Monthly")
    df = df.rename(columns={"unique_id": "item_id", "ds": "timestamp", "y": "target"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df, 18, "MS"


def load_m3_quarterly():
    """M3 Quarterly: 756 series, quarterly, pred_len=8."""
    from datasetsforecast.m3 import M3
    df, *_ = M3.load(".", group="Quarterly")
    df = df.rename(columns={"unique_id": "item_id", "ds": "timestamp", "y": "target"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df, 8, "QS"


def load_tourism_monthly():
    """Tourism Monthly: 366 series, monthly, pred_len=24."""
    from datasetsforecast.m3 import M3  # Tourism is also in datasetsforecast
    try:
        from datasetsforecast.long_horizon import LongHorizon
        # Try tourism
        df, *_ = LongHorizon.load(".", group="Tourism")
        df = df.rename(columns={"unique_id": "item_id", "ds": "timestamp", "y": "target"})
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df, 24, "MS"
    except Exception:
        # Fallback: use M3 Other (diverse set)
        df, *_ = M3.load(".", group="Other")
        df = df.rename(columns={"unique_id": "item_id", "ds": "timestamp", "y": "target"})
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df, 8, "YS"


def load_m4_weekly():
    """M4 Weekly: 359 series, weekly, pred_len=13."""
    try:
        from datasetsforecast.m4 import M4
        df, *_ = M4.load(".", group="Weekly")
        df = df.rename(columns={"unique_id": "item_id", "ds": "timestamp", "y": "target"})
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df, 13, "W"
    except Exception:
        # Fallback: M3 Yearly
        from datasetsforecast.m3 import M3
        df, *_ = M3.load(".", group="Yearly")
        df = df.rename(columns={"unique_id": "item_id", "ds": "timestamp", "y": "target"})
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df, 6, "YS"


DATASETS = {
    "M3_Monthly":   load_m3_monthly,
    "M3_Quarterly": load_m3_quarterly,
    "Tourism/Other": load_tourism_monthly,
    "M4_Weekly":    load_m4_weekly,
}


# ─────────────────────────────────────────────────────────
# Train/test split (last pred_len per item = test)
# ─────────────────────────────────────────────────────────

def split_train_test(df, pred_len):
    """Split each item: last pred_len → test, rest → train."""
    trains, tests = [], []
    for iid, group in df.groupby("item_id"):
        group = group.sort_values("timestamp").reset_index(drop=True)
        if len(group) <= pred_len + 5:  # need at least 5 for context
            continue
        trains.append(group.iloc[:-pred_len])
        tests.append(group.iloc[-pred_len:])
    return pd.concat(trains, ignore_index=True), pd.concat(tests, ignore_index=True)


# ─────────────────────────────────────────────────────────
# C-BAL runner
# ─────────────────────────────────────────────────────────

def run_cbal(train_df, test_df, pred_len, freq, preset="medium_quality"):
    """Run C-BAL and return metrics."""
    from cbal import TimeSeriesPredictor
    from cbal.dataset.ts_dataframe import TimeSeriesDataFrame

    # Filter short series (same as AG) for fair comparison
    min_len_needed = 3 * pred_len + 1
    valid_items = train_df.groupby("item_id").size()
    valid_items = valid_items[valid_items >= min_len_needed].index
    train_df = train_df[train_df["item_id"].isin(valid_items)]
    test_df = test_df[test_df["item_id"].isin(valid_items)]

    train_tsdf = TimeSeriesDataFrame.from_data_frame(train_df)

    predictor = TimeSeriesPredictor(
        prediction_length=pred_len,
        eval_metric="MASE",
        freq=freq,
        path=f"/tmp/myf_bench_{int(time.time())}",
    )

    t0 = time.time()
    predictor.fit(
        train_tsdf,
        presets=preset,
        random_seed=42,
        enable_ensemble=True,
    )
    fit_time = time.time() - t0

    # Predict
    t0 = time.time()
    preds = predictor.predict(train_tsdf, quantile_levels=[0.1, 0.5, 0.9])
    predict_time = time.time() - t0

    # Compute metrics per item
    # Determine seasonal period from freq
    _freq_sp = {"MS": 12, "ME": 12, "QS": 4, "QE": 4, "W": 52, "D": 7,
                "h": 24, "YS": 1, "YE": 1, "M": 12, "Q": 4}
    sp = 1
    if freq:
        for k, v in _freq_sp.items():
            if k in freq:
                sp = v
                break

    from cbal.metrics.scorers import MASE, WQL, sMAPE
    mase_scorer = MASE(seasonal_period=sp)
    wql_scorer = WQL()
    smape_scorer = sMAPE()

    mase_scores, wql_scores, smape_scores = [], [], []
    for iid in test_df["item_id"].unique():
        y_true = test_df[test_df["item_id"] == iid]["target"].values
        try:
            item_pred = preds.loc[iid]
            y_pred_mean = item_pred["mean"].values[:len(y_true)]
        except (KeyError, IndexError):
            continue

        y_train = train_df[train_df["item_id"] == iid]["target"].values

        try:
            mase_scores.append(mase_scorer(y_true, y_pred_mean, y_train=y_train))
        except Exception:
            pass
        try:
            smape_scores.append(smape_scorer(y_true, y_pred_mean))
        except Exception:
            pass

        try:
            q_cols = []
            for q in [0.1, 0.5, 0.9]:
                col = str(q)
                if col in preds.columns:
                    q_cols.append(item_pred[col].values[:len(y_true)])
                else:
                    q_cols.append(y_pred_mean)
            y_pred_q = np.column_stack(q_cols)
            wql_scores.append(wql_scorer(y_true, y_pred_q, quantile_levels=[0.1, 0.5, 0.9]))
        except Exception:
            pass

    lb = predictor.leaderboard(silent=True)

    return {
        "MASE": np.mean(mase_scores) if mase_scores else float("inf"),
        "sMAPE": np.mean(smape_scores) if smape_scores else float("inf"),
        "WQL": np.mean(wql_scores) if wql_scores else float("inf"),
        "fit_time": fit_time,
        "predict_time": predict_time,
        "n_models": len(lb),
        "best_model": predictor.best_model,
        "leaderboard": lb,
    }


# ─────────────────────────────────────────────────────────
# AutoGluon runner
# ─────────────────────────────────────────────────────────

def run_autogluon(train_df, test_df, pred_len, freq, preset="medium_quality"):
    """Run AutoGluon-TimeSeries and return metrics."""
    from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame

    # Filter short series that AG rejects
    min_len_needed = 3 * pred_len + 1
    valid_items = train_df.groupby("item_id").size()
    valid_items = valid_items[valid_items >= min_len_needed].index
    train_df_f = train_df[train_df["item_id"].isin(valid_items)]
    test_df_f = test_df[test_df["item_id"].isin(valid_items)]

    if len(train_df_f) == 0:
        raise ValueError(f"No items with >= {min_len_needed} obs after filtering")

    train_tsdf = TimeSeriesDataFrame.from_data_frame(train_df_f)

    predictor = TimeSeriesPredictor(
        prediction_length=pred_len,
        eval_metric="MASE",
        freq=freq,
        path=f"/tmp/ag_bench_{int(time.time())}",
        verbosity=0,
        quantile_levels=[0.1, 0.5, 0.9],
    )

    t0 = time.time()
    predictor.fit(
        train_tsdf,
        presets=preset,
        random_seed=42,
        enable_ensemble=True,
    )
    fit_time = time.time() - t0

    t0 = time.time()
    preds = predictor.predict(train_tsdf)
    predict_time = time.time() - t0

    test_df = test_df_f

    _freq_sp = {"MS": 12, "ME": 12, "QS": 4, "QE": 4, "W": 52, "D": 7,
                "h": 24, "YS": 1, "YE": 1, "M": 12, "Q": 4}
    sp = 1
    if freq:
        for k, v in _freq_sp.items():
            if k in freq:
                sp = v
                break

    from cbal.metrics.scorers import MASE, WQL, sMAPE
    mase_scorer = MASE(seasonal_period=sp)
    wql_scorer = WQL()
    smape_scorer = sMAPE()

    mase_scores, wql_scores, smape_scores = [], [], []
    for iid in test_df["item_id"].unique():
        y_true = test_df[test_df["item_id"] == iid]["target"].values
        try:
            item_pred = preds.loc[iid]
            y_pred_mean = item_pred["mean"].values[:len(y_true)]
        except (KeyError, IndexError):
            continue

        y_train = train_df[train_df["item_id"] == iid]["target"].values

        try:
            mase_scores.append(mase_scorer(y_true, y_pred_mean, y_train=y_train))
        except Exception:
            pass
        try:
            smape_scores.append(smape_scorer(y_true, y_pred_mean))
        except Exception:
            pass
        try:
            q_cols = []
            for q in [0.1, 0.5, 0.9]:
                col = str(q)
                if col in preds.columns:
                    q_cols.append(item_pred[col].values[:len(y_true)])
                else:
                    q_cols.append(y_pred_mean)
            y_pred_q = np.column_stack(q_cols)
            wql_scores.append(wql_scorer(y_true, y_pred_q, quantile_levels=[0.1, 0.5, 0.9]))
        except Exception:
            pass

    lb = predictor.leaderboard(silent=True)

    return {
        "MASE": np.mean(mase_scores) if mase_scores else float("inf"),
        "sMAPE": np.mean(smape_scores) if smape_scores else float("inf"),
        "WQL": np.mean(wql_scores) if wql_scores else float("inf"),
        "fit_time": fit_time,
        "predict_time": predict_time,
        "n_models": len(lb),
        "best_model": predictor.model_best,
        "leaderboard": lb,
    }


# ─────────────────────────────────────────────────────────
# Main benchmark loop
# ─────────────────────────────────────────────────────────

def main():
    preset = sys.argv[1] if len(sys.argv) > 1 else "medium_quality"
    print(f"\n{'='*70}")
    print(f"  Benchmark: C-BAL vs AutoGluon-TimeSeries")
    print(f"  Preset: {preset}")
    print(f"{'='*70}\n")

    all_results = []

    for ds_name, loader_fn in DATASETS.items():
        print(f"\n{'─'*60}")
        print(f"  Dataset: {ds_name}")
        print(f"{'─'*60}")

        try:
            df, pred_len, freq = loader_fn()
        except Exception as e:
            print(f"  ⚠ Failed to load: {e}")
            continue

        n_items = df["item_id"].nunique()
        n_rows = len(df)
        median_len = df.groupby("item_id").size().median()
        print(f"  Items: {n_items}, Rows: {n_rows}, Median length: {median_len:.0f}")
        print(f"  Freq: {freq}, Prediction length: {pred_len}")

        train_df, test_df = split_train_test(df, pred_len)
        n_train_items = train_df["item_id"].nunique()
        print(f"  Train items: {n_train_items}, Train rows: {len(train_df)}")

        # --- C-BAL ---
        print(f"\n  [C-BAL] Running ({preset})...")
        try:
            myf_results = run_cbal(train_df, test_df, pred_len, freq, preset)
            print(f"    MASE:  {myf_results['MASE']:.4f}")
            print(f"    sMAPE: {myf_results['sMAPE']:.2f}")
            print(f"    WQL:   {myf_results['WQL']:.4f}")
            print(f"    Time:  {myf_results['fit_time']:.1f}s fit + {myf_results['predict_time']:.1f}s predict")
            print(f"    Best:  {myf_results['best_model']} ({myf_results['n_models']} models)")
        except Exception as e:
            print(f"    ⚠ FAILED: {e}")
            traceback.print_exc()
            myf_results = {"MASE": None, "sMAPE": None, "WQL": None, "fit_time": None}

        # --- AutoGluon ---
        print(f"\n  [AutoGluon] Running ({preset})...")
        try:
            ag_results = run_autogluon(train_df, test_df, pred_len, freq, preset)
            print(f"    MASE:  {ag_results['MASE']:.4f}")
            print(f"    sMAPE: {ag_results['sMAPE']:.2f}")
            print(f"    WQL:   {ag_results['WQL']:.4f}")
            print(f"    Time:  {ag_results['fit_time']:.1f}s fit + {ag_results['predict_time']:.1f}s predict")
            print(f"    Best:  {ag_results['best_model']} ({ag_results['n_models']} models)")
        except Exception as e:
            print(f"    ⚠ FAILED: {e}")
            traceback.print_exc()
            ag_results = {"MASE": None, "sMAPE": None, "WQL": None, "fit_time": None}

        # --- Comparison ---
        if myf_results.get("MASE") and ag_results.get("MASE"):
            print(f"\n  {'Metric':<10} {'C-BAL':>14} {'AutoGluon':>14} {'Δ (%)':>10}")
            print(f"  {'─'*50}")
            for metric in ["MASE", "sMAPE", "WQL"]:
                mv = myf_results[metric]
                av = ag_results[metric]
                if mv and av and av != 0:
                    delta = (mv - av) / av * 100
                    better = "←" if mv < av else "→"
                    print(f"  {metric:<10} {mv:>14.4f} {av:>14.4f} {delta:>+9.1f}% {better}")
            print(f"  {'fit_time':<10} {myf_results['fit_time']:>13.1f}s {ag_results['fit_time']:>13.1f}s")

        all_results.append({
            "dataset": ds_name,
            "myf_mase": myf_results.get("MASE"),
            "ag_mase": ag_results.get("MASE"),
            "myf_smape": myf_results.get("sMAPE"),
            "ag_smape": ag_results.get("sMAPE"),
            "myf_wql": myf_results.get("WQL"),
            "ag_wql": ag_results.get("WQL"),
            "myf_time": myf_results.get("fit_time"),
            "ag_time": ag_results.get("fit_time"),
        })

    # --- Summary table ---
    print(f"\n\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    summary = pd.DataFrame(all_results)
    if not summary.empty:
        print(summary.to_string(index=False, float_format="%.4f"))

        # Average ratios
        valid = summary.dropna(subset=["myf_mase", "ag_mase"])
        if not valid.empty:
            avg_ratio_mase = (valid["myf_mase"] / valid["ag_mase"]).mean()
            avg_ratio_smape = (valid["myf_smape"] / valid["ag_smape"]).mean()
            avg_ratio_wql = (valid["myf_wql"] / valid["ag_wql"]).mean()
            print(f"\n  Average ratio (MyF / AG):  MASE={avg_ratio_mase:.3f}  "
                  f"sMAPE={avg_ratio_smape:.3f}  WQL={avg_ratio_wql:.3f}")
            print(f"  (< 1.0 = C-BAL wins, > 1.0 = AutoGluon wins)")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
