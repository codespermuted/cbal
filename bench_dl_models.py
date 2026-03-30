#!/usr/bin/env python3
"""
DL Model-by-Model Benchmark: C-BAL vs AutoGluon — Multiple datasets
====================================================================

Tests individual DL models AND ensemble across multiple datasets to identify
which C-BAL models beat/lose to AG counterparts.

Usage:
    python bench_dl_models.py                    # All datasets, all models
    python bench_dl_models.py ETTh1 ETTh2        # Specific datasets
    python bench_dl_models.py --ensemble-only     # Only ensemble comparison
"""
import sys, time, warnings, traceback, os
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

DATA_DIR = "/tmp/bench_data"

# ── Dataset loaders ──
def _load_ett(name, freq):
    fpath = f"{DATA_DIR}/{name}.csv"
    if not os.path.exists(fpath):
        os.makedirs(DATA_DIR, exist_ok=True)
        url = f"https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/{name}.csv"
        print(f"  Downloading {name}...")
        df = pd.read_csv(url)
        df.to_csv(fpath, index=False)
    else:
        df = pd.read_csv(fpath)
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
        if len(g) <= pl + 10:
            continue
        tr.append(g.iloc[:-pl])
        te.append(g.iloc[-pl:])
    return pd.concat(tr, ignore_index=True), pd.concat(te, ignore_index=True)

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred[:len(y_true)])))

def calc_metrics(preds, test_df, train_df):
    """Calculate MAE per item, return mean."""
    maes = []
    for iid in test_df["item_id"].unique():
        yt = test_df[test_df["item_id"] == iid]["target"].values
        try:
            yp = preds.loc[iid]["mean"].values[:len(yt)]
        except:
            continue
        maes.append(float(np.mean(np.abs(yt - yp))))
    return np.mean(maes) if maes else None

# ── C-BAL individual model runner ──
def run_cbal_model(model_name, train_df, test_df, pred_len, freq, hp=None):
    from cbal import TimeSeriesPredictor
    from cbal.dataset.ts_dataframe import TimeSeriesDataFrame
    hp = hp or {}
    tsdf = TimeSeriesDataFrame.from_data_frame(train_df)
    p = TimeSeriesPredictor(
        prediction_length=pred_len, eval_metric="MAE", freq=freq,
        path=f"/tmp/cbal_dl_{model_name}_{int(time.time())}")
    t0 = time.time()
    p.fit(tsdf, presets={"models": {model_name: hp}, "ensemble": "SimpleAverage"},
          enable_ensemble=False, random_seed=42)
    ft = time.time() - t0
    preds = p.predict(tsdf)
    m = calc_metrics(preds, test_df, train_df)
    return m, ft

# ── AG individual model runner ──
def run_ag_model(model_name, train_df, test_df, pred_len, freq, hp=None):
    from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
    hp = hp or {}
    # Filter short items
    min_l = 3 * pred_len + 1
    valid = train_df.groupby("item_id").size()
    valid = valid[valid >= min_l].index
    tdf = train_df[train_df["item_id"].isin(valid)]
    tedf = test_df[test_df["item_id"].isin(valid)]
    if len(tdf) == 0:
        raise ValueError("No valid items for AG")
    tsdf = TimeSeriesDataFrame.from_data_frame(tdf)
    p = TimeSeriesPredictor(
        prediction_length=pred_len, eval_metric="MAE", freq=freq,
        path=f"/tmp/ag_dl_{model_name}_{int(time.time())}", verbosity=0)
    t0 = time.time()
    p.fit(tsdf, hyperparameters={model_name: hp}, enable_ensemble=False, random_seed=42)
    ft = time.time() - t0
    preds = p.predict(tsdf)
    m = calc_metrics(preds, tedf, tdf)
    return m, ft

# ── Ensemble runner ──
def run_cbal_ensemble(train_df, test_df, pred_len, freq, preset="medium_quality"):
    from cbal import TimeSeriesPredictor
    from cbal.dataset.ts_dataframe import TimeSeriesDataFrame
    tsdf = TimeSeriesDataFrame.from_data_frame(train_df)
    p = TimeSeriesPredictor(
        prediction_length=pred_len, eval_metric="MAE", freq=freq,
        path=f"/tmp/cbal_ens_{int(time.time())}")
    t0 = time.time()
    p.fit(tsdf, presets=preset, random_seed=42, enable_ensemble=True)
    ft = time.time() - t0
    preds = p.predict(tsdf)
    m = calc_metrics(preds, test_df, train_df)
    return m, ft, getattr(p, 'best_model', 'ensemble')

def run_ag_ensemble(train_df, test_df, pred_len, freq, preset="medium_quality"):
    from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
    min_l = 3 * pred_len + 1
    valid = train_df.groupby("item_id").size()
    valid = valid[valid >= min_l].index
    tdf = train_df[train_df["item_id"].isin(valid)]
    tedf = test_df[test_df["item_id"].isin(valid)]
    if len(tdf) == 0:
        raise ValueError("No valid items")
    tsdf = TimeSeriesDataFrame.from_data_frame(tdf)
    p = TimeSeriesPredictor(
        prediction_length=pred_len, eval_metric="MAE", freq=freq,
        path=f"/tmp/ag_ens_{int(time.time())}", verbosity=0)
    t0 = time.time()
    p.fit(tsdf, presets=preset, random_seed=42, enable_ensemble=True)
    ft = time.time() - t0
    preds = p.predict(tsdf)
    m = calc_metrics(preds, tedf, tdf)
    return m, ft, p.model_best

# ── Model configurations ──
# (display_name, ag_name, ag_hp, cbal_name, cbal_hp)
DL_MODELS = [
    ("PatchTST",  "PatchTST", {},
                   "PatchTST", {"max_epochs": 50, "d_model": 128, "n_layers": 3,
                                "learning_rate": 1e-3, "patience": 20, "batch_size": 64}),
    ("DLinear",   "DLinear", {},
                  "DLinear", {"max_epochs": 50, "learning_rate": 1e-3, "patience": 20}),
    ("DeepAR",    "DeepAR", {},
                  "DeepAR", {"max_epochs": 50, "hidden_size": 40,
                             "learning_rate": 1e-3, "patience": 20, "batch_size": 64}),
    ("TFT",       "TemporalFusionTransformer", {},
                  "TFT", {"max_epochs": 50, "d_model": 64, "patience": 20,
                           "batch_size": 64, "learning_rate": 1e-3}),
    ("N-HiTS",    "DirectTabular", {},   # AG doesn't have N-HiTS; compare vs DirectTabular
                  "N-HiTS", {"max_epochs": 50, "hidden_size": 256,
                             "learning_rate": 1e-3, "patience": 20}),
]

BASELINE_MODELS = [
    ("SeasonalNaive", "SeasonalNaive", {}, "SeasonalNaive", {}),
    ("RecursiveTab",  "RecursiveTabular", {}, "RecursiveTabular", {"backend": "LightGBM"}),
]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]

    ensemble_only = "--ensemble-only" in flags
    include_baseline = "--with-baseline" in flags or not args

    ds_names = args if args else list(DATASETS.keys())

    print(f"\n{'='*90}")
    print(f"  DL Model Benchmark: C-BAL vs AutoGluon")
    print(f"  Datasets: {ds_names}")
    print(f"  Changes: MAE loss (was MSE), PatchTST patch_stride fix, DLinear lr=1e-3")
    print(f"{'='*90}")

    all_results = []

    for ds_name in ds_names:
        if ds_name not in DATASETS:
            print(f"\n  {ds_name}: UNKNOWN dataset, skipping")
            continue
        try:
            df, pl, freq = DATASETS[ds_name]()
        except Exception as e:
            print(f"\n  {ds_name}: SKIP ({e})")
            continue

        ni = df["item_id"].nunique()
        ml = int(df.groupby("item_id").size().median())
        train_df, test_df = split(df, pl)

        print(f"\n{'─'*90}")
        print(f"  {ds_name} | items={ni} median_len={ml} pred_len={pl} freq={freq}")
        print(f"{'─'*90}")

        if not ensemble_only:
            models = DL_MODELS + (BASELINE_MODELS if include_baseline else [])
            print(f"\n  {'Model':<16} {'C-BAL MAE':>12} {'AG MAE':>12} {'Δ%':>8} {'Winner':>8} {'C-BAL t':>8} {'AG t':>8}")
            print(f"  {'─'*72}")

            for display, ag_name, ag_hp, cbal_name, cbal_hp in models:
                # C-BAL
                try:
                    cbal_mae, cbal_t = run_cbal_model(cbal_name, train_df, test_df, pl, freq, cbal_hp)
                    cbal_str = f"{cbal_mae:.4f}" if cbal_mae else "FAIL"
                except Exception as e:
                    cbal_mae, cbal_t = None, 0
                    cbal_str = "FAIL"
                    print(f"    ! C-BAL {display}: {e}")

                # AG
                try:
                    ag_mae, ag_t = run_ag_model(ag_name, train_df, test_df, pl, freq, ag_hp)
                    ag_str = f"{ag_mae:.4f}" if ag_mae else "FAIL"
                except Exception as e:
                    ag_mae, ag_t = None, 0
                    ag_str = "FAIL"
                    print(f"    ! AG {display}: {e}")

                delta = ""
                winner = ""
                if cbal_mae and ag_mae:
                    d = (cbal_mae - ag_mae) / ag_mae * 100
                    delta = f"{d:+.1f}%"
                    winner = "C-BAL" if d < 0 else "AG"

                print(f"  {display:<16} {cbal_str:>12} {ag_str:>12} {delta:>8} {winner:>8} {cbal_t:>7.1f}s {ag_t:>7.1f}s")
                all_results.append({
                    "dataset": ds_name, "type": "individual", "model": display,
                    "cbal_mae": cbal_mae, "ag_mae": ag_mae,
                    "cbal_time": cbal_t, "ag_time": ag_t,
                })

        # ── Ensemble comparison ──
        print(f"\n  --- Ensemble (medium_quality) ---")
        try:
            cbal_ens_mae, cbal_ens_t, cbal_best = run_cbal_ensemble(train_df, test_df, pl, freq)
            cbal_ens_str = f"{cbal_ens_mae:.4f}" if cbal_ens_mae else "FAIL"
        except Exception as e:
            cbal_ens_mae, cbal_ens_t, cbal_best = None, 0, "FAIL"
            cbal_ens_str = "FAIL"
            print(f"    ! C-BAL ensemble: {e}")
            traceback.print_exc()

        try:
            ag_ens_mae, ag_ens_t, ag_best = run_ag_ensemble(train_df, test_df, pl, freq)
            ag_ens_str = f"{ag_ens_mae:.4f}" if ag_ens_mae else "FAIL"
        except Exception as e:
            ag_ens_mae, ag_ens_t, ag_best = None, 0, "FAIL"
            ag_ens_str = "FAIL"
            print(f"    ! AG ensemble: {e}")

        ens_delta = ""
        ens_winner = ""
        if cbal_ens_mae and ag_ens_mae:
            d = (cbal_ens_mae - ag_ens_mae) / ag_ens_mae * 100
            ens_delta = f"{d:+.1f}%"
            ens_winner = "C-BAL" if d < 0 else "AG"

        print(f"  {'ENSEMBLE':<16} {cbal_ens_str:>12} {ag_ens_str:>12} {ens_delta:>8} {ens_winner:>8} {cbal_ens_t:>7.0f}s {ag_ens_t:>7.0f}s")
        print(f"  C-BAL best: {cbal_best} | AG best: {ag_best}")

        all_results.append({
            "dataset": ds_name, "type": "ensemble", "model": "ENSEMBLE",
            "cbal_mae": cbal_ens_mae, "ag_mae": ag_ens_mae,
            "cbal_time": cbal_ens_t, "ag_time": ag_ens_t,
        })

    # ── Summary ──
    print(f"\n\n{'='*90}")
    print("  SUMMARY")
    print(f"{'='*90}")

    rdf = pd.DataFrame(all_results)
    if rdf.empty:
        print("  No results.")
        return

    # Individual models summary
    indiv = rdf[rdf["type"] == "individual"].dropna(subset=["cbal_mae", "ag_mae"])
    if not indiv.empty:
        print(f"\n  Individual Model Results:")
        print(f"  {'Dataset':<12} {'Model':<16} {'C-BAL':>10} {'AG':>10} {'Δ%':>8}")
        print(f"  {'─'*56}")
        for _, r in indiv.iterrows():
            d = (r["cbal_mae"] - r["ag_mae"]) / r["ag_mae"] * 100
            mark = "✓" if d < 0 else ""
            print(f"  {r['dataset']:<12} {r['model']:<16} {r['cbal_mae']:>10.4f} {r['ag_mae']:>10.4f} {d:>+7.1f}% {mark}")

        wins = (indiv["cbal_mae"] < indiv["ag_mae"]).sum()
        total = len(indiv)
        avg_ratio = (indiv["cbal_mae"] / indiv["ag_mae"]).mean()
        print(f"\n  Individual: C-BAL wins {wins}/{total} | Avg ratio: {avg_ratio:.3f}")

    # Ensemble summary
    ens = rdf[rdf["type"] == "ensemble"].dropna(subset=["cbal_mae", "ag_mae"])
    if not ens.empty:
        print(f"\n  Ensemble Results:")
        print(f"  {'Dataset':<12} {'C-BAL':>10} {'AG':>10} {'Δ%':>8}")
        print(f"  {'─'*42}")
        for _, r in ens.iterrows():
            d = (r["cbal_mae"] - r["ag_mae"]) / r["ag_mae"] * 100
            mark = "✓" if d < 0 else ""
            print(f"  {r['dataset']:<12} {r['cbal_mae']:>10.4f} {r['ag_mae']:>10.4f} {d:>+7.1f}% {mark}")

        ens_wins = (ens["cbal_mae"] < ens["ag_mae"]).sum()
        ens_total = len(ens)
        ens_ratio = (ens["cbal_mae"] / ens["ag_mae"]).mean()
        print(f"\n  Ensemble: C-BAL wins {ens_wins}/{ens_total} | Avg ratio: {ens_ratio:.3f}")

    print(f"\n{'='*90}\n")


if __name__ == "__main__":
    main()
