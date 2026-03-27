#!/usr/bin/env python3
"""
Full Benchmark: MyForecaster vs AutoGluon — Multiple presets × datasets
======================================================================
"""
import sys, time, warnings, traceback, numpy as np, pandas as pd
warnings.filterwarnings("ignore")

DATA_DIR = "/tmp/bench_data"

# ── Datasets ──
def _load_ett(name, freq):
    df = pd.read_csv(f"{DATA_DIR}/{name}.csv")
    df["date"] = pd.to_datetime(df["date"])
    return pd.DataFrame({"item_id": name, "timestamp": df["date"], "target": df["OT"].astype(float)}), 96, freq

def _load_exchange():
    from gluonts.dataset.repository import get_dataset
    from pathlib import Path
    ds = get_dataset("exchange_rate", path=Path(f"{DATA_DIR}/gluonts"))
    rows = []
    for i, e in enumerate(ds.train):
        start = e["start"].to_timestamp()
        t = e["target"]
        dates = pd.date_range(start=start, periods=len(t), freq="B")
        for d, v in zip(dates, t):
            rows.append({"item_id": f"cur_{i}", "timestamp": d, "target": float(v)})
    return pd.DataFrame(rows), 30, "B"

def _load_m3monthly():
    from datasetsforecast.m3 import M3
    df, *_ = M3.load(".", group="Monthly")
    df = df.rename(columns={"unique_id": "item_id", "ds": "timestamp", "y": "target"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df, 18, "MS"

DATASETS = {
    "ETTh1":     lambda: _load_ett("ETTh1", "h"),
    "ETTh2":     lambda: _load_ett("ETTh2", "h"),
    "ETTm1":     lambda: _load_ett("ETTm1", "15min"),
    "Exchange":  _load_exchange,
    "M3Monthly": _load_m3monthly,
}

def split(df, pl):
    tr, te = [], []
    for iid, g in df.groupby("item_id"):
        g = g.sort_values("timestamp").reset_index(drop=True)
        if len(g) <= pl + 10: continue
        tr.append(g.iloc[:-pl]); te.append(g.iloc[-pl:])
    return pd.concat(tr, ignore_index=True), pd.concat(te, ignore_index=True)

# ── Metrics ──
def calc_metrics(preds, test_df, train_df, freq):
    from myforecaster.metrics.scorers import MAE, RMSE, sMAPE
    mae_s, rmse_s, smape_s = MAE(), RMSE(), sMAPE()
    maes, rmses, smapes = [], [], []
    for iid in test_df["item_id"].unique():
        yt = test_df[test_df["item_id"]==iid]["target"].values
        try:
            yp = preds.loc[iid]["mean"].values[:len(yt)]
        except: continue
        maes.append(mae_s(yt, yp)); rmses.append(rmse_s(yt, yp))
        try: smapes.append(smape_s(yt, yp))
        except: pass
    return {
        "MAE": np.mean(maes) if maes else None,
        "RMSE": np.mean(rmses) if rmses else None,
        "sMAPE": np.mean(smapes) if smapes else None,
    }

# ── Runners ──
def run_myf(train_df, test_df, pl, freq, preset):
    from myforecaster import TimeSeriesPredictor
    from myforecaster.dataset.ts_dataframe import TimeSeriesDataFrame
    tsdf = TimeSeriesDataFrame.from_data_frame(train_df)
    p = TimeSeriesPredictor(prediction_length=pl, eval_metric="MAE", freq=freq,
                            path=f"/tmp/myf_{int(time.time())}")
    t0 = time.time()
    p.fit(tsdf, presets=preset, random_seed=42, enable_ensemble=True)
    ft = time.time() - t0
    preds = p.predict(tsdf)
    m = calc_metrics(preds, test_df, train_df, freq)
    m["time"] = ft; m["best"] = p.best_model
    return m

def run_ag(train_df, test_df, pl, freq, preset):
    from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
    min_l = 3*pl+1
    valid = train_df.groupby("item_id").size()
    valid = valid[valid >= min_l].index
    tdf = train_df[train_df["item_id"].isin(valid)]
    tedf = test_df[test_df["item_id"].isin(valid)]
    if len(tdf)==0: raise ValueError("No items")
    tsdf = TimeSeriesDataFrame.from_data_frame(tdf)
    p = TimeSeriesPredictor(prediction_length=pl, eval_metric="MAE", freq=freq,
                            path=f"/tmp/ag_{int(time.time())}", verbosity=0)
    t0 = time.time()
    p.fit(tsdf, presets=preset, random_seed=42, enable_ensemble=True)
    ft = time.time() - t0
    preds = p.predict(tsdf)
    m = calc_metrics(preds, tedf, tdf, freq)
    m["time"] = ft; m["best"] = p.model_best
    return m

# ── Main ──
def main():
    presets = sys.argv[1:] if len(sys.argv) > 1 else ["medium_quality"]

    print(f"\n{'='*80}")
    print(f"  Full Benchmark: MyForecaster vs AutoGluon")
    print(f"  Presets: {presets}")
    print(f"{'='*80}")

    results = []
    for ds_name, loader in DATASETS.items():
        try:
            df, pl, freq = loader()
        except Exception as e:
            print(f"\n  {ds_name}: SKIP ({e})")
            continue

        ni = df["item_id"].nunique()
        ml = int(df.groupby("item_id").size().median())
        train_df, test_df = split(df, pl)

        print(f"\n{'─'*80}")
        print(f"  {ds_name} | items={ni} median_len={ml} pred_len={pl} freq={freq}")
        print(f"{'─'*80}")

        for preset in presets:
            # MyForecaster
            try:
                myf = run_myf(train_df, test_df, pl, freq, preset)
                myf_str = f"MAE={myf['MAE']:.3f} RMSE={myf['RMSE']:.3f} t={myf['time']:.0f}s"
            except Exception as e:
                myf = {"MAE": None, "RMSE": None, "time": None, "best": "FAIL"}
                myf_str = f"FAIL: {e}"

            # AutoGluon
            try:
                ag = run_ag(train_df, test_df, pl, freq, preset)
                ag_str = f"MAE={ag['MAE']:.3f} RMSE={ag['RMSE']:.3f} t={ag['time']:.0f}s"
            except Exception as e:
                ag = {"MAE": None, "RMSE": None, "time": None, "best": "FAIL"}
                ag_str = f"FAIL: {e}"

            delta = ""
            if myf.get("MAE") and ag.get("MAE"):
                d = (myf["MAE"] - ag["MAE"]) / ag["MAE"] * 100
                w = "MyF wins!" if d < 0 else "AG wins"
                delta = f"  Δ={d:+.1f}% ({w})"

            print(f"  [{preset}] MyF: {myf_str}")
            print(f"  [{preset}]  AG: {ag_str}{delta}")

            results.append({
                "dataset": ds_name, "preset": preset,
                "myf_mae": myf.get("MAE"), "ag_mae": ag.get("MAE"),
                "myf_rmse": myf.get("RMSE"), "ag_rmse": ag.get("RMSE"),
                "myf_time": myf.get("time"), "ag_time": ag.get("time"),
                "myf_best": myf.get("best"), "ag_best": ag.get("best"),
            })

    # Summary
    print(f"\n\n{'='*80}")
    print("  SUMMARY TABLE")
    print(f"{'='*80}")
    sdf = pd.DataFrame(results)
    if not sdf.empty:
        for p in presets:
            sub = sdf[sdf["preset"]==p].copy()
            if sub.empty: continue
            print(f"\n  Preset: {p}")
            print(f"  {'Dataset':<12} {'MyF MAE':>10} {'AG MAE':>10} {'Δ%':>8} {'MyF t':>7} {'AG t':>7}")
            print(f"  {'─'*56}")
            for _, r in sub.iterrows():
                mm = f"{r['myf_mae']:.3f}" if r['myf_mae'] else "FAIL"
                am = f"{r['ag_mae']:.3f}" if r['ag_mae'] else "FAIL"
                d = f"{(r['myf_mae']-r['ag_mae'])/r['ag_mae']*100:+.1f}%" if r['myf_mae'] and r['ag_mae'] else ""
                mt = f"{r['myf_time']:.0f}s" if r['myf_time'] else ""
                at = f"{r['ag_time']:.0f}s" if r['ag_time'] else ""
                print(f"  {r['dataset']:<12} {mm:>10} {am:>10} {d:>8} {mt:>7} {at:>7}")

            v = sub.dropna(subset=["myf_mae","ag_mae"])
            if not v.empty:
                ratio = (v["myf_mae"]/v["ag_mae"]).mean()
                wins = (v["myf_mae"] < v["ag_mae"]).sum()
                print(f"  Avg ratio: {ratio:.3f} | MyF wins: {wins}/{len(v)}")
    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    main()
