import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from hmmlearn import hmm

# -----------------------------
# Minimal Streamlit UI
# -----------------------------
st.set_page_config(layout="wide")
st.title("HMM minimalista: 2 estados, log-return + rango")

ticker = st.text_input("Especie", "QQQ")
forecast_horizon = int(st.number_input("Horizonte de predicción (días hábiles)", value=15, min_value=1, step=1))

# -----------------------------
# Data: simple and robust
# -----------------------------
# Download single-ticker OHLC; keep it simple
data = yf.download(ticker, start="1990-01-01", auto_adjust=False)

if len(data) < 100:
    st.error("Muy pocos datos. Revisa el ticker o el rango temporal.")
    st.stop()

# Ensure ascending index and required columns
data = data.sort_index()
needed_cols = {"Open", "High", "Low", "Close"}
if not needed_cols.issubset(set(data.columns)):
    st.error("Faltan columnas OHLC en los datos.")
    st.stop()

# -----------------------------
# Features: minimal engineering
# -----------------------------
# 1) Raw log-return (Close)
data["logret"] = np.log(data["Close"] / data["Close"].shift(1))
# 2) Intraday range as a simple volatility proxy
data["range"] = (data["High"] / data["Low"]) - 1.0

# Drop NaNs introduced by shift
data = data.dropna()

# Feature matrix: first dim = logret for interpretability in forecasting
X = data[["logret", "range"]].to_numpy()

# -----------------------------
# HMM fit: 2 components, simple settings
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
means = pd.DataFrame(model.means_, columns=["mu_logret", "mu_range"])

# Map bull state by highest mean on log-return
bull_state = int(np.argmax(means["mu_logret"].values))

st.write(f"Estados HMM: 2 — Estado bull: {bull_state + 1}")
st.write(f"μ_logret por estado: {[f'{m:+.6f}' for m in means['mu_logret'].values]}")

# -----------------------------
# Vectorized long/flat backtest
# -----------------------------
# Idea: hold long when yesterday's state == bull_state (next-bar execution proxy)
ret = data["Close"].pct_change().fillna(0.0)
signal = (states == bull_state).astype(float).shift(1).fillna(0.0)

strat_ret = signal * ret
cum_strat = (1.0 + strat_ret).cumprod()
cum_bh = (1.0 + ret).cumprod()

ratio_vs_bh = (cum_strat.iloc[-1] / cum_bh.iloc[-1]) if len(cum_bh) > 0 else np.nan
st.write(f"HMM/B&H: {ratio_vs_bh:.2f}")

# -----------------------------
# Deterministic forecast (argmax transition)
# -----------------------------
last_date = data.index[-1]
last_close = data.loc[last_date, "Close"]

# Current state from the last timestamp; avoid iloc
current_state = int(states.loc[last_date])

# Transition and means
P = model.transmat_
mu_logret = model.means_[:, 0]  # expected log-return per state

# Build a deterministic path using argmax(P[state])
future_logrets = []
s = current_state
for _ in range(forecast_horizon):
    s = int(np.argmax(P[s]))
    future_logrets.append(mu_logret[s])

# Price propagation via exp(log-return)
pred_prices = [float(last_close)]
for r in future_logrets:
    pred_prices.append(pred_prices[-1] * np.exp(r))
pred_prices = pred_prices[1:]  # drop seed

pred_index = pd.date_range(start=last_date + pd.DateOffset(days=1),
                           periods=forecast_horizon, freq="B")
predicted = pd.Series(pred_prices, index=pred_index, name="pred_close")

# -----------------------------
# Plot: last 3000 points + forecast
# -----------------------------
fig = go.Figure()

tail_n = min(3000, len(data))
idx_tail = data.index[-tail_n:]
states_tail = states.loc[idx_tail]

colors = ["red", "green"]
for i in range(2):
    mask = (states_tail == i)
    x = idx_tail[mask]
    y = data.loc[idx_tail, "Close"][mask]
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers", name=f"Estado {i+1}",
        marker=dict(size=2), opacity=0.6
    ))

fig.add_trace(go.Scatter(
    x=predicted.index, y=predicted.values,
    mode="lines+markers", name="Pred."
))

fig.update_layout(
    title=f"{ticker} — estados HMM y predicción determinista",
    xaxis_title="Fecha",
    yaxis_title="Precio de cierre",
    template="plotly_dark",
    margin=dict(l=20, r=20, t=40, b=20)
)

st.plotly_chart(fig, use_container_width=True)
