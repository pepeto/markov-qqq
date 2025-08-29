"""
Minimal daily HMM with macro (Stooq + Yahoo VIX)
- Focus on a few high-signal features to avoid overfitting (7 features):
    1) logret              = log(C_t / C_{t-1})           [price drift; interpretable]
    2) range               = (H_t / L_t) - 1              [intraday volatility proxy]
    3) rv20                = sqrt(sum_{i=1..20} r_{t-i+1}^2)  [realized volatility]
    4) ret_20d             = log(C_t / C_{t-20})          [medium-term momentum]
    5) VIX_z60             = 60d z-score of VIX           [market stress, normalized]
    6) SLOPE_2s10s_z60     = 60d z-score of (US10Y - US2Y) [macro cycle / risk regime]
    7) dUS10Y_1d           = daily change of US10Y yield  [rates impulse]

- Data:
    * Price (target symbol): Stooq via pandas_datareader
    * Yields (US10Y, US2Y): Stooq via pandas_datareader
    * VIX: Yahoo via yfinance (Stooq often blocks commodities via DataReader)

- Model:
    * GaussianHMM(n_components=2, covariance_type="diag") for stability
    * First feature kept as raw log-return to interpret state means
    * Signal smoothing:
        - Hysteresis on P(bull): enter > 0.60, exit < 0.40
        - Minimum dwell time: >= 3 bars before adopting a new regime

- Plot:
    * y-axis logarithmic
    * green = bull regime, red = bear regime
    * deterministic N-step forecast via argmax transition path

- Style:
    * No try/except
    * No .iloc (use .loc and index labels)
    * Comments in English; UI in Spanish.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from hmmlearn import hmm
from pandas_datareader.data import DataReader
import yfinance as yf

# -----------------------------
# Streamlit UI (minimal)
# -----------------------------
st.set_page_config(layout="wide")
st.title("HMM minimalista (Stooq) con VIX y tasas")

symbol_in = st.text_input("Especie (Stooq)", "QQQ")
forecast_horizon = int(st.number_input("Horizonte de predicción (días hábiles)", value=15, min_value=1, step=1))

# Smoothing hyperparameters (fixed, simple)
PROBA_ENTER = 0.60   # enter long if P(bull) > 0.60
PROBA_EXIT  = 0.40   # exit long if P(bull) < 0.40
MIN_DWELL   = 3      # minimum bars in a regime before switching

# Normalize symbol (Stooq tickers uppercase)
SYMBOL = symbol_in.strip().upper()

# -----------------------------
# Date range: 1900 -> tomorrow (naive dates; no timezone)
# -----------------------------
start_date = pd.Timestamp(1900, 1, 1).date()
end_date = (pd.Timestamp.utcnow().floor("D") + pd.Timedelta(days=1)).date()

# -----------------------------
# Data loaders
# -----------------------------
def stooq_ohlc(symbol: str) -> pd.DataFrame:
    """Download OHLC from Stooq, ascending index."""
    df = DataReader(symbol, "stooq", start=start_date, end=end_date)
    df = df.sort_index()
    return df

def stooq_close(symbol: str) -> pd.Series:
    """Download 'Close' from Stooq, ascending index."""
    df = stooq_ohlc(symbol)
    return df["Close"].rename(symbol)

def yahoo_close(symbol: str) -> pd.Series:
    """Download 'Close' from Yahoo via yfinance, ascending index."""
    df = yf.download(symbol, start="1900-01-01", progress=False, auto_adjust=False)
    df = df.sort_index()
    s = pd.Series(df["Close"].astype("float64"))
    s.index = pd.to_datetime(s.index).tz_localize(None)  # ensure naive datetime index
    return s.rename(symbol)

# -----------------------------
# Download price and macro series
# -----------------------------
price = stooq_ohlc(SYMBOL)
needed = {"Open", "High", "Low", "Close"}
if not needed.issubset(set(price.columns)):
    st.error("Faltan columnas OHLC en los datos de Stooq para el símbolo elegido.")
    st.stop()

if len(price) < 200:
    st.error("Muy pocos datos para entrenar el HMM. Revisa el símbolo o rango temporal.")
    st.stop()

# Macro:
# - VIX from Yahoo (^VIX) because Stooq often blocks commodities through DataReader.
# - US10Y and US2Y from Stooq.
VIX   = yahoo_close("^VIX").rename("VIX")
US10Y = stooq_close("10YUSY.B").rename("US10Y")
US2Y  = stooq_close("2YUSY.B").rename("US2Y")

# -----------------------------
# Feature helpers
# -----------------------------
def zscore_roll(s: pd.Series, win: int = 60) -> pd.Series:
    """Rolling z-score with a fixed window (min_periods=win)."""
    mean = s.rolling(win, min_periods=win).mean()
    std  = s.rolling(win, min_periods=win).std(ddof=0)
    z = (s - mean) / std
    return z.rename(f"{s.name}_z{win}")

def diff1(s: pd.Series) -> pd.Series:
    """First difference."""
    return s.diff(1).rename(f"d{s.name}_1d")

def hysteresis_binary(proba: pd.Series, upper: float, lower: float) -> pd.Series:
    """
    Convert probability series to binary with hysteresis:
      - If prev==0 and proba>upper: switch to 1
      - If prev==1 and proba<lower: switch to 0
      - Else: keep previous
    """
    out = pd.Series(index=proba.index, dtype="float64")
    prev = 0.0
    for dt in proba.index:
        p = float(proba.loc[dt])
        if prev == 0.0 and p > upper:
            prev = 1.0
        elif prev == 1.0 and p < lower:
            prev = 0.0
        out.loc[dt] = prev
    return out.rename("signal_hyst")

def enforce_min_dwell(signal: pd.Series, min_bars: int) -> pd.Series:
    """
    Enforce minimum dwell time on a binary signal:
      - Confirm a regime only after 'min_bars' consecutive bars in the new state.
    """
    out = pd.Series(index=signal.index, dtype="float64")
    current = 0.0
    counter = 0
    for dt in signal.index:
        s = float(signal.loc[dt])
        if s == current:
            counter = counter + 1
        else:
            counter = 1
        if counter >= min_bars:
            current = s
        out.loc[dt] = current
    return out.rename("signal_dwell")

# -----------------------------
# Macro engineered series
# -----------------------------
slope_2s10s = (US10Y - US2Y).rename("SLOPE_2s10s")
macro = pd.concat([
    zscore_roll(VIX, 60),               # VIX_z60
    diff1(US10Y),                       # dUS10Y_1d
    zscore_roll(slope_2s10s, 60),       # SLOPE_2s10s_z60
], axis=1)

# Align price with macro on a common calendar; forward-fill macro gaps
data = price.join(macro, how="left").ffill()

# -----------------------------
# Price features (minimal)
# -----------------------------
data["logret"]   = np.log(data["Close"] / data["Close"].shift(1))
data["range"]    = (data["High"] / data["Low"]) - 1.0
data["rv20"]     = np.sqrt((data["logret"]**2).rolling(20, min_periods=20).sum())
data["ret_20d"]  = np.log(data["Close"] / data["Close"].shift(20))

# Final clean-up (drop NaNs from rolling/z-score windows)
data = data.dropna()

# -----------------------------
# Assemble HMM feature matrix (explicit order)
# -----------------------------
feature_cols = [
    "logret",            # interpretable, kept raw
    "range",             # intraday volatility proxy
    "rv20",              # realized volatility (20d)
    "ret_20d",           # medium-term momentum
    "VIX_z60",           # normalized market stress
    "SLOPE_2s10s_z60",   # normalized curve slope
    "dUS10Y_1d",         # yield impulse
]
for col in feature_cols:
    if col not in data.columns:
        st.error(f"Falta la columna requerida: {col}. Amplía el rango temporal.")
        st.stop()

X = data[feature_cols].to_numpy()

# -----------------------------
# Train HMM (2 states, diagonal covariance)
# -----------------------------
model = hmm.GaussianHMM(
    n_components=2,
    covariance_type="diag",
    n_iter=300,
    tol=1e-2,
    init_params="stmc",
    random_state=42
)
model.fit(X)

states = pd.Series(model.predict(X), index=data.index, name="state")
proba  = pd.DataFrame(model.predict_proba(X), index=data.index, columns=[f"p_state_{i}" for i in range(model.n_components)])

# Identify bull state by highest mean on first feature (logret)
mu = pd.DataFrame(model.means_, columns=feature_cols)
bull_state = int(np.argmax(mu["logret"].values))
bear_state = 1 - bull_state  # valid for K=2

st.write(f"Estados HMM: 2 — Estado alcista: {bull_state + 1} — Convergió: {model.monitor_.converged} (iter={model.monitor_.iter})")
st.write(f"μ(logret) por estado: {[f'{m:+.6f}' for m in mu['logret'].values]}")

# -----------------------------
# Smoothed trading signal (hysteresis + min dwell)
# -----------------------------
p_bull = proba[f"p_state_{bull_state}"].rename("p_bull")
sig_hyst = hysteresis_binary(p_bull, PROBA_ENTER, PROBA_EXIT)
sig_smooth = enforce_min_dwell(sig_hyst, MIN_DWELL).rename("signal")

# Execute at next bar: shift by 1 (no look-ahead)
signal_exec = sig_smooth.shift(1).fillna(0.0)

# -----------------------------
# Vectorized backtest (long/flat)
# -----------------------------
ret = data["Close"].pct_change().fillna(0.0)
strat_curve = (1.0 + signal_exec * ret).cumprod()
bh_curve = (1.0 + ret).cumprod()

last_label = strat_curve.index[-1]
ratio_vs_bh = strat_curve.loc[last_label] / bh_curve.loc[last_label]
st.write(f"HMM/B&H (señal suavizada): {ratio_vs_bh:.2f}")

# -----------------------------
# Deterministic forecast (argmax transition path)
# -----------------------------
last_date = data.index[-1]
last_close = data.loc[last_date, "Close"]
current_state = int(states.loc[last_date])

P = model.transmat_
mu_logret_states = model.means_[:, 0]  # expected log-return per state

future_logrets = []
s = current_state
for _ in range(forecast_horizon):
    s = int(np.argmax(P[s]))
    future_logrets.append(mu_logret_states[s])

pred_prices = [float(last_close)]
for r in future_logrets:
    pred_prices.append(pred_prices[-1] * np.exp(r))
pred_prices = pred_prices[1:]

pred_index = pd.date_range(start=last_date + pd.DateOffset(days=1),
                           periods=forecast_horizon, freq="B")
predicted = pd.Series(pred_prices, index=pred_index, name="pred_close")

# -----------------------------
# Plot: last 3000 points + forecast, log-y, green=bull, red=bear
# -----------------------------
fig = go.Figure()

tail_n = min(3000, len(data))
idx_tail = data.index[-tail_n:]
states_tail = states.loc[idx_tail]

state_colors = {bull_state: "green", bear_state: "red"}
state_names  = {bull_state: "Régimen alcista", bear_state: "Régimen bajista"}

# Bear first, then Bull (legend ordering)
for i in [bear_state, bull_state]:
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
    title=f"{SYMBOL} — HMM (Stooq) con VIX/tasas y predicción determinista",
    xaxis_title="Fecha",
    yaxis_title="Precio de cierre (log)",
    yaxis_type="log",
    template="plotly_dark",
    margin=dict(l=20, r=20, t=40, b=20)
)

st.plotly_chart(fig, use_container_width=True)
