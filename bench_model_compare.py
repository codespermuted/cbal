#!/usr/bin/env python3
"""
Model-by-Model Comparison: C-BAL vs AutoGluon-TimeSeries
========================================================

Compares individual models (not ensembles) on ETTh1 to identify
which C-BAL model implementations are weak vs AG.
"""
import sys, time, warnings, traceback
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

DATA_DIR = "/tmp/bench_data"

def load_etth1():
    df = pd.read_csv(f"{DATA_DIR}/ETTh1.csv")
    df["date"] = pd.to_datetime(df["date"])
    ts_df = pd.DataFrame({"item_id": "ETTh1", "timestamp": df["date"], "target": df["OT"].astype(float)})
    return ts_df, 96, "h"

def split(df, pl):
    return df.iloc[:-pl].copy(), df.iloc[-pl:].copy()

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred[:len(y_true)])))

# ── C-BAL individual model runner ──
def run_cbal_model(model_name, train_df, test_df, pred_len, freq, hp=None):
    from cbal import TimeSeriesPredictor
    from cbal.dataset.ts_dataframe import TimeSeriesDataFrame
    hp = hp or {}
    tsdf = TimeSeriesDataFrame.from_data_frame(train_df)
    p = TimeSeriesPredictor(
        prediction_length=pred_len, eval_metric="MAE", freq=freq,
        path=f"/tmp/cbal_mc_{model_name}_{int(time.time())}")
    t0 = time.time()
    # Use custom preset with ONLY this model (no default preset models)
    p.fit(tsdf, presets={"models": {model_name: hp}, "ensemble": "SimpleAverage"},
          enable_ensemble=False, random_seed=42)
    ft = time.time() - t0
    preds = p.predict(tsdf)
    y_true = test_df["target"].values
    try:
        y_pred = preds.loc["ETTh1"]["mean"].values
    except:
        y_pred = preds.iloc[:pred_len]["mean"].values
    return mae(y_true, y_pred), ft

# ── AG individual model runner ──
def run_ag_model(model_name, train_df, test_df, pred_len, freq, hp=None):
    from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
    hp = hp or {}
    tsdf = TimeSeriesDataFrame.from_data_frame(train_df)
    p = TimeSeriesPredictor(
        prediction_length=pred_len, eval_metric="MAE", freq=freq,
        path=f"/tmp/ag_mc_{model_name}_{int(time.time())}", verbosity=0)
    t0 = time.time()
    p.fit(tsdf, hyperparameters={model_name: hp}, enable_ensemble=False, random_seed=42)
    ft = time.time() - t0
    preds = p.predict(tsdf)
    y_true = test_df["target"].values
    try:
        y_pred = preds.loc["ETTh1"]["mean"].values
    except:
        y_pred = preds.iloc[:pred_len]["mean"].values
    return mae(y_true, y_pred), ft

# ── Model mapping: AG name → C-BAL name + HP ──
MODELS = [
    # (display_name, ag_name, ag_hp, cbal_name, cbal_hp)
    ("SeasonalNaive",   "SeasonalNaive", {}, "SeasonalNaive", {}),
    ("ETS",             "ETS", {},           "AutoETS", {}),
    ("Theta",           "Theta", {},         "AutoTheta", {}),
    ("RecursiveTabular","RecursiveTabular", {}, "RecursiveTabular", {"backend": "LightGBM"}),
    ("DirectTabular",   "DirectTabular", {},   "DirectTabular", {"backend": "LightGBM"}),
    ("DeepAR",          "DeepAR", {},          "DeepAR", {"max_epochs": 50, "patience": 20}),
    ("TFT",  "TemporalFusionTransformer", {},  "TFT", {"max_epochs": 50, "d_model": 64, "patience": 15}),
    ("PatchTST",        "PatchTST", {},        "PatchTST", {"max_epochs": 50, "d_model": 128, "n_layers": 3, "patience": 10}),
    ("DLinear",         "DLinear", {},         "DLinear", {"max_epochs": 50, "patience": 10}),
]

def main():
    dataset = sys.argv[1] if len(sys.argv) > 1 else "ETTh1"
    ts_df, pred_len, freq = load_etth1()
    train_df, test_df = split(ts_df, pred_len)

    print(f"\n{'='*80}")
    print(f"  Model-by-Model: C-BAL vs AutoGluon | {dataset} pred_len={pred_len}")
    print(f"{'='*80}\n")
    print(f"  {'Model':<20} {'C-BAL MAE':>12} {'AG MAE':>12} {'Δ%':>8} {'C-BAL t':>8} {'AG t':>8}")
    print(f"  {'─'*68}")

    results = []
    for display, ag_name, ag_hp, cbal_name, cbal_hp in MODELS:
        # C-BAL
        try:
            cbal_mae, cbal_t = run_cbal_model(cbal_name, train_df, test_df, pred_len, freq, cbal_hp)
            cbal_str = f"{cbal_mae:.4f}"
        except Exception as e:
            cbal_mae, cbal_t = None, 0
            cbal_str = f"FAIL"
            print(f"    C-BAL {display} error: {e}")

        # AG
        try:
            ag_mae, ag_t = run_ag_model(ag_name, train_df, test_df, pred_len, freq, ag_hp)
            ag_str = f"{ag_mae:.4f}"
        except Exception as e:
            ag_mae, ag_t = None, 0
            ag_str = f"FAIL"
            print(f"    AG {display} error: {e}")

        delta = ""
        if cbal_mae and ag_mae:
            d = (cbal_mae - ag_mae) / ag_mae * 100
            w = "<-" if d < 0 else "->"
            delta = f"{d:+.1f}%"

        print(f"  {display:<20} {cbal_str:>12} {ag_str:>12} {delta:>8} {cbal_t:>7.1f}s {ag_t:>7.1f}s")
        results.append({"model": display, "cbal_mae": cbal_mae, "ag_mae": ag_mae,
                        "cbal_time": cbal_t, "ag_time": ag_t})

    print(f"\n{'='*80}")
    valid = [r for r in results if r["cbal_mae"] and r["ag_mae"]]
    if valid:
        wins = sum(1 for r in valid if r["cbal_mae"] < r["ag_mae"])
        avg_ratio = np.mean([r["cbal_mae"]/r["ag_mae"] for r in valid])
        print(f"  C-BAL wins: {wins}/{len(valid)} | Avg ratio: {avg_ratio:.3f}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
