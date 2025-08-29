import streamlit as st
import talib as tl  # TA-Lib for technical indicators
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# -----------------------------
# Streamlit page configuration
# -----------------------------
st.set_page_config(layout="wide")
st.title('Análisis de Estados Ocultos con HMM (TA-Lib)')

# -----------------------------
# Inputs
# -----------------------------
especie = st.text_input('Especie', 'QQQ')
sma_period = st.number_input('Período de SMA', value=44, min_value=1, step=1)
ema_period = st.number_input('Período de EMA', value=30, min_value=1, step=1)
wma_period = st.number_input('Período de WMA', value=52, min_value=1, step=1)
rsi_period = st.number_input('Período de RSI', value=14, min_value=1, step=1)
willr_period = st.number_input('Período de WILLR', value=14, min_value=1, step=1)
cci_period = st.number_input('Período de CCI', value=20, min_value=1, step=1)

# -----------------------------
# Data download
# -----------------------------
# NOTE: You currently use yfinance; if desired, you can switch to pandas_datareader "stooq"
# for your usual workflow. Kept yfinance here to minimize changes around Streamlit inputs.
data = yf.download(especie, start="1970-01-01", end="2025-05-03")
st.write(f'Longitud de los datos descargados: {len(data)}')

# Defensive check
if data.empty:
    st.error("No se descargaron datos. Verifica el ticker o el rango de fechas.")
    st.stop()

# -----------------------------
# Feature construction (TA-Lib)
# -----------------------------
# We keep a single 'OHL' series (close) and compute log-returns in original scale.
data['OHL'] = data['Close']
data['Log Return'] = np.log(data['OHL'] / data['OHL'].shift(1))

# Moving averages via TA-Lib
data['SMA'] = tl.SMA(data['OHL'].to_numpy(), timeperiod=int(sma_period))
data['EMA'] = tl.EMA(data['OHL'].to_numpy(), timeperiod=int(ema_period))
data['WMA'] = tl.WMA(data['OHL'].to_numpy(), timeperiod=int(wma_period))

# Log-returns of MAs (these will be standardized individually)
data['SMA_log_return'] = np.log(data['SMA'] / data['SMA'].shift(1))
data['EMA_log_return'] = np.log(data['EMA'] / data['EMA'].shift(1))
data['WMA_log_return'] = np.log(data['WMA'] / data['WMA'].shift(1))

# Momentum/Oscillators via TA-Lib
data['RSI']   = tl.RSI(data['OHL'].to_numpy(), timeperiod=int(rsi_period))
data['WILLR'] = tl.WILLR(
    data['High'].to_numpy(),
    data['Low'].to_numpy(),
    data['Close'].to_numpy(),
    timeperiod=int(willr_period)
)
data['CCI']   = tl.CCI(
    data['High'].to_numpy(),
    data['Low'].to_numpy(),
    data['Close'].to_numpy(),
    timeperiod=int(cci_period)
)

# Intraday range (kept unscaled; you may consider log-range)
data['Range'] = (data['High'] / data['Low']) - 1.0

# Drop NaN rows produced by indicators
data = data.dropna()

# -----------------------------
# Scaling (feature-wise)
# -----------------------------
# NOTE: 'Log Return' is kept in original scale to be the first HMM dimension.
# Each scaled column uses its own scaler to avoid cross-contamination.
sma_scaler = StandardScaler()
ema_scaler = StandardScaler()
wma_scaler = StandardScaler()

rsi_scaler    = MinMaxScaler()
willr_scaler  = MinMaxScaler()
cci_scaler    = MinMaxScaler()

data['sma_scaled']   = sma_scaler.fit_transform(data[['SMA_log_return']]).ravel()
data['ema_scaled']   = ema_scaler.fit_transform(data[['EMA_log_return']]).ravel()
data['wma_scaled']   = wma_scaler.fit_transform(data[['WMA_log_return']]).ravel()

# RSI is naturally in [0, 100]; WILLR is in [-100, 0]; CCI is unbounded (often within ±200)
# We normalize each to [0, 1] for numerical stability in HMM.
data['rsi_scaled']   = rsi_scaler.fit_transform(data[['RSI']]).ravel()
data['willr_scaled'] = willr_scaler.fit_transform(data[['WILLR']]).ravel()
data['cci_scaled']   = cci_scaler.fit_transform(data[['CCI']]).ravel()

# -----------------------------
# HMM training
# -----------------------------
# Feature matrix: first column = raw log-return (important for prediction)
feature_cols = ['Log Return', 'sma_scaled', 'cci_scaled', 'willr_scaled', 'ema_scaled', 'Range']
X = data[feature_cols].to_numpy()

# HMM with reproducibility
model = hmm.GaussianHMM(
    algorithm='map',
    n_components=2,
    covariance_type='full',
    n_iter=300,
    tol=1e-2,
    init_params='stmc',
    random_state=42
)
model.fit(X)

hidden_states = model.predict(X)
log_likelihood = model.score(X)

st.write(f'Converged: {model.monitor_.converged} — Iterations: {model.monitor_.iter}')
st.write(f'Log Likelihood: {log_likelihood:.0f}')

# -----------------------------
# Backtest
# -----------------------------
def BackTest(close_series: pd.Series, states: np.ndarray, inversion: bool = False,
             comision: float = 1 - 0.04/100, capital_inicial: float = 100.0, verbose: int = 0) -> float:
    """
    Simple long/flat strategy driven by state:
    - If (state == 1) XOR inversion -> long; else -> flat.
    - Uses next day's open-price proxy = next close (since we only have close), solely for consistency with your original approach.
    """
    # Ensure alignment
    if len(close_series) != len(states):
        raise ValueError("States length must match close series length.")

    dates = close_series.index
    first_date = dates[0]
    last_date  = dates[-1]

    cash = capital_inicial
    shares = 0.0
    bought = False
    operations = 0

    # Buy & Hold benchmark
    bh_shares = capital_inicial / close_series.loc[first_date]
    bh_final_value = bh_shares * close_series.loc[last_date]

    for i in range(len(dates) - 1):
        d_now = dates[i]
        d_next = dates[i + 1]

        go_long = (states[i] == 1)
        if inversion:
            go_long = not go_long

        if go_long and not bought:
            # Buy at next bar price proxy
            price = close_series.loc[d_next]
            shares = (cash * comision) / price
            cash = 0.0
            bought = True
            operations += 1
        elif (not go_long) and bought:
            # Sell at next bar price proxy
            price = close_series.loc[d_next]
            cash = shares * price * comision
            shares = 0.0
            bought = False
            operations += 1

    final_value = cash + shares * close_series.loc[last_date]
    ratio_vs_bh = final_value / bh_final_value

    if verbose == 1:
        st.write(f'B&H final value: {bh_final_value:.2f}')
        st.write(f'Operations: {operations}')

    return ratio_vs_bh

st.write('* BackTest *')
ratio_normal = BackTest(data['Close'], hidden_states, inversion=False, verbose=1)
ratio_inverse = BackTest(data['Close'], hidden_states, inversion=True,  verbose=0)
st.write(f'HMM/B&H (mejor de long/short-invertido): {max(ratio_normal, ratio_inverse):.2f}')

# -----------------------------
# One-step-ahead "most likely" path forecast (deterministic via argmax)
# -----------------------------
transition = model.transmat_
means = model.means_       # shape: (n_components, n_features)
covars = model.covars_     # kept for completeness

current_state = hidden_states[-1]
predicted_states = [current_state]
horizon = 15

for _ in range(horizon):
    next_state = int(np.argmax(transition[current_state]))
    predicted_states.append(next_state)
    current_state = next_state

# Use the first feature mean (raw log return) for projection
predicted_log_returns = [means[s, 0] for s in predicted_states[1:]]

last_date = data.index[-1]
last_close = data.loc[last_date, 'OHL']

predicted_prices = [last_close]
for r in predicted_log_returns:
    # Price_{t+1} = Price_t * exp(log-return)
    predicted_prices.append(predicted_prices[-1] * np.exp(r))

pred_index = pd.date_range(start=last_date + pd.DateOffset(days=1),
                           periods=len(predicted_log_returns),
                           freq='B')
predicted_df = pd.DataFrame({'OHL': predicted_prices[1:]}, index=pred_index)

# -----------------------------
# Plot (last N points + forecast)
# -----------------------------
colors = ['red', 'green', 'blue', 'purple', 'orange']
fig = go.Figure()

tail = min(6000, len(data))
idx_tail = data.index[-tail:]
states_tail = hidden_states[-tail:]

for i in range(model.n_components):
    mask = (states_tail == i)
    x = idx_tail[mask]
    y = data.loc[idx_tail, 'OHL'].to_numpy()[mask]
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='markers',
        opacity=0.6,
        marker=dict(color=colors[i % len(colors)], size=2),
        name=f'Estado {i + 1}'
    ))

fig.add_trace(go.Scatter(
    x=predicted_df.index,
    y=predicted_df['OHL'],
    mode='lines+markers',
    name='Pred.',
    line=dict(color='grey')
))

fig.update_layout(
    title=f'Precio de cierre de {especie}: estados ocultos y proyección (HMM)',
    xaxis_title='Fecha',
    yaxis_title='Precio de cierre (log)',
    yaxis_type='log',
    autosize=True,
    margin=dict(l=20, r=20, t=40, b=20),
    template="plotly_dark"
)

st.plotly_chart(fig, use_container_width=True)
