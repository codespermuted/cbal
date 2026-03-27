"""
cbal 패키지 활용 예시

설치:  pip install -e ".[stats]"
실행:  python example_usage.py
차트:  pip install plotly && python example_usage.py
"""

import numpy as np
import pandas as pd

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Import
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
from cbal import TimeSeriesPredictor, TimeSeriesDataFrame

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. 목 데이터 (5매장 × 180일)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
np.random.seed(42)
stores = ["서울점", "부산점", "대구점", "인천점", "광주점"]
rows = []
for store in stores:
    base = np.random.uniform(80, 200)
    for t in range(180):
        d = pd.Timestamp("2024-01-01") + pd.Timedelta(days=t)
        dow = d.dayofweek
        val = (base + 0.12 * t
               + 15 * np.sin(2 * np.pi * dow / 7)
               - 8 * (dow >= 5)
               + np.random.randn() * 5)
        rows.append({"item_id": store, "timestamp": d, "target": round(max(0, val), 1)})

data = TimeSeriesDataFrame.from_data_frame(pd.DataFrame(rows))
train, test = data.train_test_split(prediction_length=14)
print(f"데이터: {data.num_items}매장 × 180일 = {len(data)}행\n")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Fit — 모델 학습 + 앙상블
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
predictor = TimeSeriesPredictor(prediction_length=14, eval_metric="MASE")
predictor.fit(
    train,
    hyperparameters={
        "Naive": {},
        "SeasonalNaive": {"seasonal_period": 7},
        "Average": {},
        "Drift": {},
        # 서버에서 추가 추천:
        # "AutoETS": {},
        # "AutoTheta": {},
    },
    enable_ensemble=True,
    num_val_windows=2,
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. 리더보드
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("📊 리더보드:")
lb = predictor.leaderboard(silent=True)
print(lb.to_string(index=False))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. 앙상블 가중치
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if predictor._ensemble and hasattr(predictor._ensemble, "weights"):
    print("\n🔗 앙상블 가중치:")
    for m, w in sorted(predictor._ensemble.weights.items(), key=lambda x: -x[1]):
        print(f"  {m:16s} {w:.3f} {'█' * int(w * 30)}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. 예측
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
preds = predictor.predict(train, quantile_levels=[0.1, 0.5, 0.9])
print(f"\n🔮 예측: {preds.shape}")
for s in stores:
    fp = preds.loc[s]
    print(f"  {s}: mean [{fp['mean'].min():.0f} ~ {fp['mean'].max():.0f}]  "
          f"90%PI [{fp['0.1'].min():.0f} ~ {fp['0.9'].max():.0f}]")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. Test 평가
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n📈 모델별 Test MAE:")
for mname in predictor.model_names:
    try:
        s = predictor.score(test, model=mname, metric="MAE")
        marker = " ◀ BEST" if mname == predictor.best_model else ""
        print(f"  {mname:20s} {s:6.2f}{marker}")
    except:
        pass

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. Plotly 차트 (선택)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
try:
    import plotly.graph_objects as go

    store = "서울점"
    fig = go.Figure()

    # 과거 실적
    hist = train.loc[store]["target"].iloc[-42:]
    fig.add_trace(go.Scatter(x=hist.index, y=hist.values,
                             mode="lines", line=dict(color="gray"), name="실적"))
    # 90% 예측구간
    fc = preds.loc[store]
    fig.add_trace(go.Scatter(x=fc.index, y=fc["0.9"], line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=fc.index, y=fc["0.1"], fill="tonexty",
                             fillcolor="rgba(55,138,221,0.15)", line=dict(width=0), name="90% PI"))
    # 예측 평균
    fig.add_trace(go.Scatter(x=fc.index, y=fc["mean"], mode="lines",
                             line=dict(color="#378ADD", width=2.5), name="예측"))
    # 실제값
    actual = test.loc[store]["target"].iloc[-14:]
    fig.add_trace(go.Scatter(x=actual.index, y=actual.values, mode="markers",
                             marker=dict(color="red", size=5), name="실제"))

    fig.update_layout(title=f"{store} 14일 수요 예측 ({predictor.best_model})",
                      template="plotly_white", height=400)
    fig.write_html("forecast_result.html")
    print("\n✅ forecast_result.html 저장! 브라우저에서 열어보세요.")
except ImportError:
    print("\n💡 pip install plotly 후 재실행하면 차트가 생성됩니다.")

print("\n완료!")
