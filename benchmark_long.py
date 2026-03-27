#!/usr/bin/env python3
"""
Long-horizon Benchmark: C-BAL vs AutoGluon-TimeSeries
============================================================

Informer/Autoformer-style benchmarks:
- ETTh1, ETTh2 (hourly, single long series, pred_len=96)
- ETTm1, ETTm2 (15min, single long series, pred_len=96)
- Exchange Rate (daily, 8 series, pred_len=96)
- Electricity (hourly, 321 series, pred_len=96)

These are LONG-HORIZON forecasting tasks — very different from M3/M4.
"""

import sys
import time
import warnings
import traceback

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA_DIR = "/tmp/bench_data"

# ─────────────────────────────────────────────────────────
# Dataset loaders
# ─────────────────────────────────────────────────────────

def load_ett(name, freq, pred_len=96):
    """Load ETT dataset (single multivariate → split into per-column univariate)."""
    df = pd.read_csv(f"{DATA_DIR}/{name}.csv")
    df["date"] = pd.to_datetime(df["date"])
    # Target: OT (Oil Temperature) — standard in ETT benchmarks
    target_col = "OT"
    ts_df = pd.DataFrame({
        "item_id": name,
        "timestamp": df["date"],
        "target": df[target_col].values.astype(float),
    })
    return ts_df, pred_len, freq


def load_exchange():
    """Load Exchange Rate from gluonts (8 series, daily)."""
    from gluonts.dataset.repository import get_dataset
    from pathlib import Path
    ds = get_dataset("exchange_rate", path=Path(f"{DATA_DIR}/gluonts"))
    rows = []
    for i, entry in enumerate(ds.train):
        start = entry["start"].to_timestamp()
        target = entry["target"]
        dates = pd.date_range(start=start, periods=len(target), freq="B")
        for t, v in zip(dates, target):
            rows.append({"item_id": f"currency_{i}", "timestamp": t, "target": float(v)})
    return pd.DataFrame(rows), 96, "B"


def load_electricity():
    """Load Electricity from gluonts (321 series, hourly). Use subset for speed."""
    from gluonts.dataset.repository import get_dataset
    from pathlib import Path
    ds = get_dataset("electricity", path=Path(f"{DATA_DIR}/gluonts"))
    rows = []
    for i, entry in enumerate(list(ds.train)[:50]):  # 50 series for speed
        start = entry["start"].to_timestamp()
        target = entry["target"][-2000:]  # last 2000 points for speed
        dates = pd.date_range(start=start, periods=len(entry["target"]), freq="h")[-2000:]
        for t, v in zip(dates, target):
            rows.append({"item_id": f"client_{i}", "timestamp": t, "target": float(v)})
    return pd.DataFrame(rows), 96, "h"


DATASETS = {
    "ETTh1":       lambda: load_ett("ETTh1", "h", 96),
    "ETTh2":       lambda: load_ett("ETTh2", "h", 96),
    "ETTm1":       lambda: load_ett("ETTm1", "15min", 96),
    "ETTm2":       lambda: load_ett("ETTm2", "15min", 96),
    "Exchange":    load_exchange,
    "Electricity": load_electricity,
}


# ─────────────────────────────────────────────────────────
# Split
# ─────────────────────────────────────────────────────────

def split_train_test(df, pred_len):
    trains, tests = [], []
    for iid, group in df.groupby("item_id"):
        group = group.sort_values("timestamp").reset_index(drop=True)
        if len(group) <= pred_len + 10:
            continue
        trains.append(group.iloc[:-pred_len])
        tests.append(group.iloc[-pred_len:])
    return pd.concat(trains, ignore_index=True), pd.concat(tests, ignore_index=True)


# ─────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────

def compute_metrics(preds, test_df, train_df, freq):
    from cbal.metrics.scorers import MAE, RMSE, sMAPE
    mae_scorer, rmse_scorer, smape_scorer = MAE(), RMSE(), sMAPE()

    mae_scores, rmse_scores, smape_scores = [], [], []
    for iid in test_df["item_id"].unique():
        y_true = test_df[test_df["item_id"] == iid]["target"].values
        try:
            item_pred = preds.loc[iid]
            y_pred = item_pred["mean"].values[:len(y_true)]
        except (KeyError, IndexError):
            continue
        mae_scores.append(mae_scorer(y_true, y_pred))
        rmse_scores.append(rmse_scorer(y_true, y_pred))
        try:
            smape_scores.append(smape_scorer(y_true, y_pred))
        except:
            pass

    return {
        "MAE": np.mean(mae_scores) if mae_scores else float("inf"),
        "RMSE": np.mean(rmse_scores) if rmse_scores else float("inf"),
        "sMAPE": np.mean(smape_scores) if smape_scores else float("inf"),
    }


# ─────────────────────────────────────────────────────────
# Runners
# ─────────────────────────────────────────────────────────

def run_cbal(train_df, test_df, pred_len, freq, preset="medium_quality"):
    from cbal import TimeSeriesPredictor
    from cbal.dataset.ts_dataframe import TimeSeriesDataFrame
    train_tsdf = TimeSeriesDataFrame.from_data_frame(train_df)

    predictor = TimeSeriesPredictor(
        prediction_length=pred_len, eval_metric="MAE", freq=freq,
        path=f"/tmp/myf_long_{int(time.time())}",
    )
    t0 = time.time()
    predictor.fit(train_tsdf, presets=preset, random_seed=42, enable_ensemble=True)
    fit_time = time.time() - t0

    t0 = time.time()
    preds = predictor.predict(train_tsdf)
    predict_time = time.time() - t0

    metrics = compute_metrics(preds, test_df, train_df, freq)
    metrics.update({"fit_time": fit_time, "predict_time": predict_time,
                    "best_model": predictor.best_model,
                    "n_models": len(predictor.model_names)})
    return metrics


def run_autogluon(train_df, test_df, pred_len, freq, preset="medium_quality"):
    from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame

    min_len = 3 * pred_len + 1
    valid = train_df.groupby("item_id").size()
    valid = valid[valid >= min_len].index
    train_df = train_df[train_df["item_id"].isin(valid)]
    test_df = test_df[test_df["item_id"].isin(valid)]
    if len(train_df) == 0:
        raise ValueError("No items after filtering")

    train_tsdf = TimeSeriesDataFrame.from_data_frame(train_df)

    predictor = TimeSeriesPredictor(
        prediction_length=pred_len, eval_metric="MAE", freq=freq,
        path=f"/tmp/ag_long_{int(time.time())}", verbosity=0,
    )
    t0 = time.time()
    predictor.fit(train_tsdf, presets=preset, random_seed=42, enable_ensemble=True)
    fit_time = time.time() - t0

    t0 = time.time()
    preds = predictor.predict(train_tsdf)
    predict_time = time.time() - t0

    metrics = compute_metrics(preds, test_df, train_df, freq)
    metrics.update({"fit_time": fit_time, "predict_time": predict_time,
                    "best_model": predictor.model_best,
                    "n_models": len(predictor.leaderboard(silent=True))})
    return metrics


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────

def main():
    preset = sys.argv[1] if len(sys.argv) > 1 else "medium_quality"
    print(f"\n{'='*70}")
    print(f"  Long-Horizon Benchmark: C-BAL vs AutoGluon")
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
            print(f"  SKIP: {e}")
            continue

        n_items = df["item_id"].nunique()
        n_rows = len(df)
        median_len = int(df.groupby("item_id").size().median())
        print(f"  Items: {n_items}, Rows: {n_rows}, Median len: {median_len}")
        print(f"  Freq: {freq}, Pred len: {pred_len}")

        train_df, test_df = split_train_test(df, pred_len)

        # C-BAL
        print(f"\n  [C-BAL] ...")
        try:
            myf = run_cbal(train_df, test_df, pred_len, freq, preset)
            print(f"    MAE={myf['MAE']:.4f}  RMSE={myf['RMSE']:.4f}  sMAPE={myf['sMAPE']:.2f}")
            print(f"    Time: {myf['fit_time']:.1f}s  Best: {myf['best_model']}")
        except Exception as e:
            print(f"    FAILED: {e}")
            traceback.print_exc()
            myf = {"MAE": None, "RMSE": None, "sMAPE": None, "fit_time": None}

        # AutoGluon
        print(f"\n  [AutoGluon] ...")
        try:
            ag = run_autogluon(train_df, test_df, pred_len, freq, preset)
            print(f"    MAE={ag['MAE']:.4f}  RMSE={ag['RMSE']:.4f}  sMAPE={ag['sMAPE']:.2f}")
            print(f"    Time: {ag['fit_time']:.1f}s  Best: {ag['best_model']}")
        except Exception as e:
            print(f"    FAILED: {e}")
            traceback.print_exc()
            ag = {"MAE": None, "RMSE": None, "sMAPE": None, "fit_time": None}

        # Compare
        if myf.get("MAE") and ag.get("MAE"):
            print(f"\n  {'Metric':<8} {'MyF':>10} {'AG':>10} {'Δ%':>8}")
            print(f"  {'─'*38}")
            for m in ["MAE", "RMSE", "sMAPE"]:
                mv, av = myf[m], ag[m]
                if mv and av and av != 0:
                    d = (mv - av) / av * 100
                    w = "←" if mv < av else "→"
                    print(f"  {m:<8} {mv:>10.4f} {av:>10.4f} {d:>+7.1f}% {w}")
            print(f"  {'time':<8} {myf['fit_time']:>9.1f}s {ag['fit_time']:>9.1f}s")

        all_results.append({
            "dataset": ds_name,
            "myf_mae": myf.get("MAE"), "ag_mae": ag.get("MAE"),
            "myf_rmse": myf.get("RMSE"), "ag_rmse": ag.get("RMSE"),
            "myf_smape": myf.get("sMAPE"), "ag_smape": ag.get("sMAPE"),
            "myf_time": myf.get("fit_time"), "ag_time": ag.get("fit_time"),
        })

    # Summary
    print(f"\n\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    summary = pd.DataFrame(all_results)
    if not summary.empty:
        print(summary.to_string(index=False, float_format="%.4f"))
        valid = summary.dropna(subset=["myf_mae", "ag_mae"])
        if not valid.empty:
            r_mae = (valid["myf_mae"] / valid["ag_mae"]).mean()
            r_rmse = (valid["myf_rmse"] / valid["ag_rmse"]).mean()
            print(f"\n  Avg ratio (MyF/AG): MAE={r_mae:.3f}  RMSE={r_rmse:.3f}")
            print(f"  (< 1.0 = C-BAL wins)")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
