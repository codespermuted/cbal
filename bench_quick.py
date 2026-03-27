#!/usr/bin/env python3
"""Quick parallel benchmark: ETTh1 only, MyF vs AG side by side."""
import sys, time, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")

DATA_DIR = "/tmp/bench_data"

def load_etth1():
    df = pd.read_csv(f"{DATA_DIR}/ETTh1.csv")
    df["date"] = pd.to_datetime(df["date"])
    return pd.DataFrame({"item_id": "ETTh1", "timestamp": df["date"], "target": df["OT"].astype(float)})

def split(df, pred_len):
    return df.iloc[:-pred_len].copy(), df.iloc[-pred_len:].copy()

def metrics(preds, test_df, train_df):
    from myforecaster.metrics.scorers import MAE, RMSE
    y_true = test_df["target"].values
    try:
        y_pred = preds.loc["ETTh1"]["mean"].values[:len(y_true)]
    except:
        y_pred = preds.iloc[:len(y_true)]["mean"].values
    return {"MAE": float(MAE()(y_true, y_pred)), "RMSE": float(RMSE()(y_true, y_pred))}

def run_myf(train_df, pred_len):
    from myforecaster import TimeSeriesPredictor
    from myforecaster.dataset.ts_dataframe import TimeSeriesDataFrame
    tsdf = TimeSeriesDataFrame.from_data_frame(train_df)
    p = TimeSeriesPredictor(prediction_length=pred_len, eval_metric="MAE", freq="h",
                            path=f"/tmp/myf_q_{int(time.time())}")
    t0 = time.time()
    p.fit(tsdf, presets="medium_quality", random_seed=42)
    fit_time = time.time() - t0
    preds = p.predict(tsdf)
    return preds, fit_time, p.best_model

def run_ag(train_df, pred_len):
    from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
    tsdf = TimeSeriesDataFrame.from_data_frame(train_df)
    p = TimeSeriesPredictor(prediction_length=pred_len, eval_metric="MAE", freq="h",
                            path=f"/tmp/ag_q_{int(time.time())}", verbosity=0)
    t0 = time.time()
    p.fit(tsdf, presets="medium_quality", random_seed=42)
    fit_time = time.time() - t0
    preds = p.predict(tsdf)
    return preds, fit_time, p.model_best

if __name__ == "__main__":
    df = load_etth1()
    pred_len = 96
    train_df, test_df = split(df, pred_len)

    runner = sys.argv[1] if len(sys.argv) > 1 else "both"

    if runner in ("myf", "both"):
        print("[MyForecaster] ETTh1 pred_len=96...")
        preds, ft, best = run_myf(train_df, pred_len)
        m = metrics(preds, test_df, train_df)
        print(f"  MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}  Time={ft:.1f}s  Best={best}")

    if runner in ("ag", "both"):
        print("[AutoGluon] ETTh1 pred_len=96...")
        preds, ft, best = run_ag(train_df, pred_len)
        m = metrics(preds, test_df, train_df)
        print(f"  MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}  Time={ft:.1f}s  Best={best}")
