import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from hmmlearn import hmm
from pandas_datareader.data import DataReader

# -----------------------------
# Minimal UI
# -----------------------------
st.set_page_config(layout="wide")
st.title("HMM minimalista con Stooq (OHLC, 2 estados)")

ticker = st.text_input("Especie", "QQQ")
forecast_horizon = int(st.number_input("Horizonte de predicción (días hábiles)", value=15, min_value=1, step=1))

# -----------------------------
# Fechas: desde 1900 hasta mañana (Buenos Aires)
# -----------------------------
tz = "America/Argentina/Buenos_Aires"
start_date = pd.Timestamp("1900-01-01")
end_date = (pd.Timestamp.now(tz=tz) + pd.Timedelta(days=1)).normalize()

# -----------------------------
# Descarga Stooq y normalización
# -----------------------------
data = DataReader(ticker, "stooq", start=start_date, end=end_date)

# Asegurar orden ascendente y nombres estándar
data = data.sort_index()
data = data.rename(columns=lambda c: c.strip().capitalize())

needed = {"Open", "High", "Low", "Close"}
if not needed.issubset(set(data.columns)):
    st.error("Faltan columnas OHLC en los datos de Stooq.")
    st.stop()

if len(data) < 100:
    st.error("Muy pocos datos. Revisa el ticker o el rango temporal.")
    st.stop()

# -----------------------------
# Features (mínimo)
# -----------------------------
data["logret"] = np.log(data["Close"] / data["Close"].shift(1))
data["range"] = (data["High"] / data["Low"]) - 1.0
data = data.dropna()

X = data[["logret", "range"]].to_numpy()

# -----------------------------
# HMM (2 estados)
# -----------------------------
model = hmm.GaussianHMM(
    n_components=2,
    covariance_type="full",
    n_iter=200,
    tol=1e-2,
    init_params="stmc",
    random_state=42
)
model.fit(X)

states = pd.Series(model.predict(X), index=data.index, name="state")
mu = pd.DataFrame(model.means_, columns=["mu_logret", "mu_range"])
bull_state = int(np.argmax(mu["mu_logret"].values))

st.write(f"Estados: 2 — Estado bull: {bull_state + 1}")
st.write(f"μ_logret por estado: {[f'{m:+.6f}' for m in mu['mu_logret'].values]}")

# -----------------------------
# Backtest vectorizado (long/flat)
# -----------------------------
ret = data["Close"].pct_change().fillna(0.0)
signal = (states == bull_state).astype(float).shift(1).fillna(0.0)

strat_curve = (1.0 + (signal * ret)).cumprod()
bh_curve = (1.0 + ret).cumprod()

last_label = strat_curve.index[-1]
ratio_vs_bh = strat_curve.loc[last_label] / bh_curve.loc[last_label]
st.write(f"HMM/B&H: {ratio_vs_bh:.2f}")

# -----------------------------
# Predicción determinista (argmax transición)
# -----------------------------
last_date = data.index[-1]
last_close = data.loc[last_date, "Close"]
current_state = int(states.loc[last_date])

P = model.transmat_
mu_logret = model.means_[:, 0]

future_logrets = []
s = current_state
for _ in range(forecast_horizon):
    s = int(np.argmax(P[s]))
    future_logrets.append(mu_logret[s])

pred_prices = [float(last_close)]
for r in future_logrets:
    pred_prices.append(pred_prices[-1] * np.exp(r))
pred_prices = pred_prices[1:]

pred_index = pd.date_range(start=last_date + pd.DateOffset(days=1),
                           periods=forecast_horizon, freq="B")
predicted = pd.Series(pred_prices, index=pred_index, name="pred_close")

# -----------------------------
# Gráfico: últimos 3000 + predicción
# -----------------------------
fig = go.Figure()

tail_n = min(3000, len(data))
idx_tail = data.index[-tail_n:]
states_tail = states.loc[idx_tail]

for i, color in zip([0, 1], ["red", "green"]):
    mask = (states_tail == i)
    x = idx_tail[mask]
    y = data.loc[idx_tail, "Close"][mask]
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers", name=f"Estado {i+1}",
        marker=dict(size=2, color=color), opacity=0.6
    ))

fig.add_trace(go.Scatter(
    x=predicted.index, y=predicted.values,
    mode="lines+markers", name="Pred."
))

fig.update_layout(
    title=f"{ticker} — HMM (Stooq) y predicción determinista",
    xaxis_title="Fecha",
    yaxis_title="Precio de cierre",
    template="plotly_dark",
    margin=dict(l=20, r=20, t=40, b=20)
)

st.plotly_chart(fig, use_container_width=True)
