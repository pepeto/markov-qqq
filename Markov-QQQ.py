"""
HMM minimalista (Stooq, daily) — ensemble 5 semillas (mediana), covarianza full
- Features:
    1) logret = log(C_t / C_{t-1})          [interpretable drift]
    2) range  = (H_t / L_t) - 1             [intraday volatility proxy]
- Modelo:
    GaussianHMM(n_components=2, covariance_type="full")
    Ensemble: seeds = [11, 17, 23, 42, 73]; combinar con mediana de P(bull)
    Bull = estado con mayor media en logret
- Backtest:
    Señal long/flat = (median P(bull) > 0.5), ejecución a barra siguiente (shift +1)
- Plot:
    Eje Y logarítmico, verde=alcista, rojo=bajista (etiquetas del mejor modelo)
- Estilo:
    Sin try/except, sin iloc. Comentarios en inglés; UI en español.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from hmmlearn import hmm
from pandas_datareader.data import DataReader

# -----------------------------
# UI mínima
# -----------------------------
st.set_page_config(layout="wide")
st.title("HMM (Stooq) — ensemble 5 seeds, covariance full")

ticker_in = st.text_input("Species (Stooq)", "QQQ")
forecast_horizon = int(st.number_input("Prediction (days)", value=15, min_value=1, step=1))

SYMBOL = ticker_in.strip().upper()

# -----------------------------
# Fechas naive para Stooq
# -----------------------------
start_date = pd.Timestamp(1900, 1, 1).date()
end_date = (pd.Timestamp.utcnow().floor("D") + pd.Timedelta(days=1)).date()

# -----------------------------
# Descarga de OHLC (Stooq)
# -----------------------------
def stooq_ohlc(symbol: str) -> pd.DataFrame:
    """Download OHLC from Stooq, ascending index."""
    df = DataReader(symbol, "stooq", start=start_date, end=end_date)
    df = df.sort_index()
    return df

data = stooq_ohlc(SYMBOL)
needed = {"Open", "High", "Low", "Close"}
if not needed.issubset(set(data.columns)):
    st.error("Faltan columnas OHLC en los datos de Stooq para el símbolo.")
    st.stop()

if len(data) < 200:
    st.error("Too few data to train HMM.")
    st.stop()

# -----------------------------
# Features mínimas
# -----------------------------
data["logret"] = np.log(data["Close"] / data["Close"].shift(1))
data["range"]  = (data["High"] / data["Low"]) - 1.0
data = data.dropna()

X = data[["logret", "range"]].to_numpy()

# -----------------------------
# Ensemble de semillas (mediana de P(bull))
# -----------------------------
seeds = [11, 17, 23, 42, 73]

p_bull_list = []
ll_list = []
models = []
bull_states = []

for rs in seeds:
    m = hmm.GaussianHMM(
        n_components=2,
        covariance_type="full",
        n_iter=200,
        tol=1e-2,
        init_params="stmc",
        random_state=int(rs)
    )
    m.fit(X)

    mu = m.means_[:, 0]  # mean of logret per state
    bull = int(np.argmax(mu))
    bear = 1 - bull

    # Descarta semillas sin polaridad (bull mean <= bear mean)
    if mu[bull] <= mu[bear]:
        continue

    p_bull = m.predict_proba(X)[:, bull]
    p_bull_list.append(pd.Series(p_bull, index=data.index))
    ll_list.append(m.score(X))
    models.append(m)
    bull_states.append(bull)

# Fallback si ninguna semilla superó el filtro (muy raro)
if len(p_bull_list) == 0:
    m = hmm.GaussianHMM(
        n_components=2,
        covariance_type="full",
        n_iter=200,
        tol=1e-2,
        init_params="stmc",
        random_state=42
    )
    m.fit(X)
    mu = m.means_[:, 0]
    bull = int(np.argmax(mu))
    p_bull_med = pd.Series(m.predict_proba(X)[:, bull], index=data.index).rename("p_bull_med")
    best_model = m
    best_bull = bull
else:
    # Mediana de probabilidades P(bull) en el tiempo
    p_bull_med = pd.concat(p_bull_list, axis=1).median(axis=1).rename("p_bull_med")
    # Mejor modelo (para plot de estados y forecast) por máximo log-likelihood
    best_idx = int(np.argmax(ll_list))
    best_model = models[best_idx]
    best_bull = bull_states[best_idx]

# -----------------------------
# Señal y backtest (long/flat) — ejecución a próxima barra
# -----------------------------
signal = (p_bull_med > 0.5).astype(float).rename("signal")
signal_exec = signal.shift(1).fillna(0.0)

ret = data["Close"].pct_change().fillna(0.0)
strat_curve = (1.0 + signal_exec * ret).cumprod()
bh_curve = (1.0 + ret).cumprod()

last_label = strat_curve.index[-1]
ratio_vs_bh = strat_curve.loc[last_label] / bh_curve.loc[last_label]
st.write(f"HMM/B&H (ensemble 5 semillas, mediana): {ratio_vs_bh:.2f}")

# -----------------------------
# Estados y forecast usando el "mejor" modelo
# -----------------------------
states_best = pd.Series(best_model.predict(X), index=data.index, name="state_best")

P = best_model.transmat_
mu_logret_states = best_model.means_[:, 0]
current_state = int(states_best.loc[data.index[-1]])

future_logrets = []
s = current_state
for _ in range(forecast_horizon):
    s = int(np.argmax(P[s]))
    future_logrets.append(mu_logret_states[s])

last_date = data.index[-1]
last_close = data.loc[last_date, "Close"]

pred_prices = [float(last_close)]
for r in future_logrets:
    pred_prices.append(pred_prices[-1] * np.exp(r))
pred_prices = pred_prices[1:]

pred_index = pd.date_range(start=last_date + pd.DateOffset(days=1),
                           periods=forecast_horizon, freq="B")
predicted = pd.Series(pred_prices, index=pred_index, name="pred_close")

# -----------------------------
# Gráfico (últimos 3000 + pred), Y log, verde/bull rojo/bear
# -----------------------------
fig = go.Figure()

tail_n = min(1000, len(data))
idx_tail = data.index[-tail_n:]
states_tail = states_best.loc[idx_tail]

bear_state = 1 - best_bull
state_colors = {best_bull: "green", bear_state: "red"}
state_names  = {best_bull: "Régimen alcista", bear_state: "Régimen bajista"}

for i in [bear_state, best_bull]:
    mask = (states_tail == i)
    x = idx_tail[mask]
    y = data.loc[idx_tail, "Close"][mask]
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers",
        name=state_names[i],
        marker=dict(size=2, color=state_colors[i]),
        opacity=0.6
    ))

fig.add_trace(go.Scatter(
    x=predicted.index, y=predicted.values,
    mode="lines+markers", name="Pred."
))

fig.update_layout(
    title=f"{SYMBOL} — HMM & prediction",
    xaxis_title="Fecha",
    yaxis_title="Precio de cierre (log)",
    yaxis_type="log",
    template="plotly_dark",
    margin=dict(l=20, r=20, t=40, b=20)
)

st.plotly_chart(fig, use_container_width=True)

