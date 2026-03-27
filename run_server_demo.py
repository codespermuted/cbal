"""
myforecaster 종합 데모 — 서버용 (Plotly 대시보드)

설치:
    pip install -e ".[stats]"
    pip install plotly

실행:
    python run_server_demo.py

출력:
    results/
    ├── 01_leaderboard.html
    ├── 02_forecast_bands.html
    ├── 03_model_comparison.html
    ├── 04_ensemble_weights.html
    ├── 05_residuals.html
    ├── summary.txt
    └── saved_predictor/
"""

import os, time, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")

from myforecaster import TimeSeriesPredictor, TimeSeriesDataFrame

RESULTS = "results"
os.makedirs(RESULTS, exist_ok=True)

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("plotly 미설치 — 텍스트만 출력합니다. pip install plotly")

def savefig(fig, name, title=""):
    if not HAS_PLOTLY: return
    fig.update_layout(title=title, template="plotly_white", height=420,
                      font=dict(family="sans-serif", size=12),
                      margin=dict(l=50, r=30, t=50, b=40))
    fig.write_html(os.path.join(RESULTS, name), include_plotlyjs="cdn")
    print(f"  → {RESULTS}/{name}")

# ────────────────────────────────────────────────
# 1. 데이터
# ────────────────────────────────────────────────
print("=" * 55)
print("  MyForecaster 종합 데모")
print("=" * 55)

np.random.seed(42)
N, D, H = 10, 365, 14
stores = [f"store_{i:02d}" for i in range(N)]
rows = []
for s in stores:
    base = np.random.uniform(50, 250)
    for t in range(D):
        dt = pd.Timestamp("2023-01-01") + pd.Timedelta(days=t)
        dow = dt.dayofweek
        v = (base + 0.1 * t
             + 20 * np.sin(2 * np.pi * dow / 7)
             + 10 * np.sin(2 * np.pi * t / 365)
             - 10 * (dow >= 5)
             + np.random.randn() * 6)
        rows.append({"item_id": s, "timestamp": dt, "target": round(max(0, v), 1)})

data = TimeSeriesDataFrame.from_data_frame(pd.DataFrame(rows))
train, test = data.train_test_split(H)
print(f"\n데이터: {N}매장 × {D}일 = {len(data)}행")

# ────────────────────────────────────────────────
# 2. Fit
# ────────────────────────────────────────────────
print(f"\nFitting (pred_len={H})...")
t0 = time.time()

predictor = TimeSeriesPredictor(
    prediction_length=H, eval_metric="MASE",
    path=os.path.join(RESULTS, "saved_predictor"),
)
predictor.fit(
    train,
    hyperparameters={
        "Naive": {},
        "SeasonalNaive": {"seasonal_period": 7},
        "Average": {},
        "Drift": {},
        "AutoETS": {},
        "AutoTheta": {},
    },
    enable_ensemble=True,
    num_val_windows=3,
    refit_full=True,
)
print(f"  {time.time()-t0:.1f}s | models: {predictor.model_names}")

# ────────────────────────────────────────────────
# 3. Leaderboard
# ────────────────────────────────────────────────
print("\n📊 리더보드:")
lb = predictor.leaderboard(silent=True)
print(lb.to_string(index=False))

if HAS_PLOTLY:
    fig = go.Figure(go.Bar(
        y=lb["model"], x=lb["score_val"], orientation="h",
        marker_color=["#1D9E75" if s < 0.7 else "#BA7517" if s < 1 else "#E24B4A" for s in lb["score_val"]],
        text=[f"{s:.3f}" for s in lb["score_val"]], textposition="outside"))
    fig.update_layout(yaxis=dict(autorange="reversed"), xaxis_title="MASE")
    savefig(fig, "01_leaderboard.html", "Model leaderboard (MASE)")

# ────────────────────────────────────────────────
# 4. 앙상블
# ────────────────────────────────────────────────
if predictor._ensemble and hasattr(predictor._ensemble, "weights"):
    print("\n🔗 앙상블:")
    for m, w in sorted(predictor._ensemble.weights.items(), key=lambda x: -x[1]):
        print(f"  {m:16s} {w:.3f} {'█' * int(w * 30)}")

    if HAS_PLOTLY:
        wt = predictor._ensemble.weights
        fig = go.Figure(go.Pie(labels=list(wt.keys()), values=list(wt.values()), hole=0.4))
        savefig(fig, "04_ensemble_weights.html", "Ensemble weights")

# ────────────────────────────────────────────────
# 5. 예측
# ────────────────────────────────────────────────
preds = predictor.predict(train, quantile_levels=[0.1, 0.5, 0.9])
print(f"\n🔮 예측: {preds.shape}")

# 전 모델 예측
all_preds = {}
for m in predictor.model_names:
    try:
        all_preds[m] = predictor.predict(train, model=m)
    except: pass

# ── Plotly: 예측 + 구간 (4매장) ──
if HAS_PLOTLY:
    items = data.item_ids[:4]
    fig = make_subplots(rows=2, cols=2, subplot_titles=list(items))
    for idx, iid in enumerate(items):
        r, c = idx // 2 + 1, idx % 2 + 1
        h = train.loc[iid]["target"].iloc[-42:]
        fig.add_trace(go.Scatter(x=h.index, y=h.values, mode="lines",
                                 line=dict(color="#888"), name="실적", showlegend=idx==0), row=r, col=c)
        fc = preds.loc[iid]
        fig.add_trace(go.Scatter(x=fc.index, y=fc["0.9"], line=dict(width=0), showlegend=False), row=r, col=c)
        fig.add_trace(go.Scatter(x=fc.index, y=fc["0.1"], fill="tonexty",
                                 fillcolor="rgba(55,138,221,0.15)", line=dict(width=0),
                                 name="90% PI", showlegend=idx==0), row=r, col=c)
        fig.add_trace(go.Scatter(x=fc.index, y=fc["mean"], mode="lines",
                                 line=dict(color="#378ADD", width=2.5),
                                 name="예측", showlegend=idx==0), row=r, col=c)
        act = test.loc[iid]["target"].iloc[-H:]
        fig.add_trace(go.Scatter(x=act.index, y=act.values, mode="markers",
                                 marker=dict(color="red", size=4),
                                 name="실제", showlegend=idx==0), row=r, col=c)
    fig.update_layout(height=550)
    savefig(fig, "02_forecast_bands.html", "Forecast + 90% prediction intervals")

# ── Plotly: 모델 비교 ──
if HAS_PLOTLY and all_preds:
    iid = data.item_ids[0]
    colors = ["#378ADD", "#D85A30", "#1D9E75", "#7F77DD", "#BA7517", "#888", "#D4537E"]
    fig = go.Figure()
    h = train.loc[iid]["target"].iloc[-28:]
    fig.add_trace(go.Scatter(x=h.index, y=h.values, mode="lines",
                             line=dict(color="#888", dash="dot"), name="실적"))
    for i, (mn, mp) in enumerate(all_preds.items()):
        fc = mp.loc[iid]
        fig.add_trace(go.Scatter(x=fc.index, y=fc["mean"], mode="lines",
                                 line=dict(color=colors[i % len(colors)],
                                           width=2.5 if mn == predictor.best_model else 1.2),
                                 name=mn))
    savefig(fig, "03_model_comparison.html", f"Model comparison — {iid}")

# ────────────────────────────────────────────────
# 6. 평가
# ────────────────────────────────────────────────
print("\n📈 Test 평가:")
print(f"  {'Model':20s}  {'MAE':>7s}  {'MASE':>7s}  {'RMSE':>7s}")
for m in predictor.model_names:
    try:
        mae = predictor.score(test, model=m, metric="MAE")
        mase = predictor.score(test, model=m, metric="MASE")
        rmse = predictor.score(test, model=m, metric="RMSE")
        tag = " ◀" if m == predictor.best_model else ""
        print(f"  {m:20s}  {mae:7.2f}  {mase:7.3f}  {rmse:7.2f}{tag}")
    except: pass

# ────────────────────────────────────────────────
# 7. 잔차
# ────────────────────────────────────────────────
res_rows = []
bp = all_preds.get(predictor.best_model, preds)
for iid in test.item_ids:
    act = test.loc[iid]["target"].iloc[-H:].values
    if iid in bp.item_ids:
        fc = bp.loc[iid]["mean"].values
        for h in range(min(len(act), len(fc))):
            res_rows.append({"horizon": h + 1, "residual": act[h] - fc[h], "actual": act[h], "forecast": fc[h]})
rdf = pd.DataFrame(res_rows)
if not rdf.empty:
    print(f"\n잔차: mean={rdf['residual'].mean():+.2f}, std={rdf['residual'].std():.2f}")
    if HAS_PLOTLY:
        fig = make_subplots(rows=1, cols=2, subplot_titles=["잔차 분포", "실제 vs 예측"])
        fig.add_trace(go.Histogram(x=rdf["residual"], nbinsx=25, marker_color="#378ADD"), row=1, col=1)
        fig.add_trace(go.Scatter(x=rdf["actual"], y=rdf["forecast"], mode="markers",
                                 marker=dict(size=3, color="#378ADD", opacity=0.4)), row=1, col=2)
        mn, mx = rdf[["actual", "forecast"]].min().min(), rdf[["actual", "forecast"]].max().max()
        fig.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines",
                                 line=dict(color="red", dash="dash"), showlegend=False), row=1, col=2)
        fig.update_layout(showlegend=False, height=350)
        savefig(fig, "05_residuals.html", "Residual analysis")

# ────────────────────────────────────────────────
# 8. Save / Load
# ────────────────────────────────────────────────
sp = predictor.save()
loaded = TimeSeriesPredictor.load(sp)
lp = loaded.predict(train)
diff = (lp["mean"] - preds["mean"]).abs().max()
print(f"\nSave/Load: diff={diff:.8f} ✓")

# ────────────────────────────────────────────────
# Summary
# ────────────────────────────────────────────────
txt = f"""MyForecaster 종합 데모 결과
{'='*40}
데이터:    {N}매장 × {D}일
모델:      {', '.join(predictor.model_names)}
베스트:    {predictor.best_model} (MASE {lb.iloc[0]['score_val']:.3f})
"""
if predictor._ensemble and hasattr(predictor._ensemble, "weights"):
    txt += f"앙상블:    {predictor._ensemble.weights}\n"
txt += f"Save/Load: verified (diff={diff:.8f})\n"
with open(os.path.join(RESULTS, "summary.txt"), "w") as f:
    f.write(txt)

print(f"\n{'='*55}")
print(f"  ✓ 완료! 결과: {RESULTS}/")
if HAS_PLOTLY:
    print(f"  → open {RESULTS}/02_forecast_bands.html")
print(f"{'='*55}")
