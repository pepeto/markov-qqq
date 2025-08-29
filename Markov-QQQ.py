"""
Streamlit HMM (TA-Lib) app
- Indicators computed with TA-Lib (EMA, WMA, RSI, WILLR, CCI, SMA).
- Clean numeric inputs for TA-Lib (float64, C-contiguous) to avoid wrapper errors.
- HMM on features with first dimension = raw log return (interpretable for forecasting).
- Simple long/flat backtest with next-bar execution proxy (using next close).
- No try/except blocks as requested. Comments concise and in English.
"""

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import talib as tl

from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# -----------------------------
# Streamlit page configuration
# -----------------------------
st.set_page_config(layout="wide")
st.title('Análisis de Estados Ocultos con HMM (TA-Lib)')

# -----------------------------
# Helpers
# -----------------------------
def to_np_f64(series: pd.Series) -> np.ndarray:
    """Coerce to numeric, cast to float64, and force C-contiguous layout for TA-Lib."""
    arr = pd.to_numeric(series, errors='coerce').astype('float64').to_numpy()
    return np.ascontiguousarray(arr)


def compute_indicators(df: pd.DataFrame,
                       sma_period: int,
                       ema_period: int,
                       wma_period: int,
                       rsi_period: int,
                       willr_period: int,
                       cci_period: int) -> pd.DataFrame:
    """Compute TA-Lib indicators and engineered features. Drops NaNs at the end."""
    # Base arrays for TA-Lib (float64, contiguous)
    open_np  = to_np_f64(df['Open'])
    high_np  = to_np_f64(df['High'])
    low_np   = to_np_f64(df['Low'])
    close_np = to_np_f64(df['Close'])

    # Base series
    df['OHL'] = df['Close']
    df['Log Return'] = np.log(df['OHL'] / df['OHL'].shift(1))

    # Moving averages via TA-Lib
    df['SMA'] = tl.SMA(close_np, timeperiod=int(sma_period))
    df['EMA'] = tl.EMA(close_np, timeperiod=int(ema_period))
    df['WMA'] = tl.WMA(close_np, timeperiod=int(wma_period))

    # Log-returns of MAs (to capture MA dynamics)
    df['SMA_log_return'] = np.log(df['SMA'] / df['SMA'].shift(1))
    df['EMA_log_return'] = np.log(df['EMA'] / df['EMA'].shift(1))
    df['WMA_log_return'] = np.log(df['WMA'] / df['WMA'].shift(1))

    # Oscillators via TA-Lib
    df['RSI']   = tl.RSI(close_np, timeperiod=int(rsi_period))
    df['WILLR'] = tl.WILLR(high_np, low_np, close_np, timeperiod=int(willr_period))  # [-100, 0]
    df['CCI']   = tl.CCI(high_np, low_np, close_np, timeperiod=int(cci_period))

    # Intraday range (optionally replace by log-range if desired)
    df['Range'] = (df['High'] / df['Low']) - 1.0

    # Replace infs and drop NaNs introduced by indicators
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """Scale selected features; keep raw log return as the first HMM dimension."""
    sma_scaler   = StandardScaler()
    ema_scaler   = StandardScaler()
    wma_scaler   = StandardScaler()
    rsi_scaler   = MinMaxScaler()
    willr_scaler = MinMaxScaler()
    cci_scaler   = MinMaxScaler()

    df['sma_scaled']   = sma_scaler.fit_transform(df[['SMA_log_return']]).ravel()
    df['ema_scaled']   = ema_scaler.fit_transform(df[['EMA_log_return']]).ravel()
    df['wma_scaled']   = wma_scaler.fit_transform(df[['WMA_log_return']]).ravel()
    df['rsi_scaled']   = rsi_scaler.fit_transform(df[['RSI']]).ravel()
    df['willr_scaled'] = willr_scaler.fit_transform(df[['WILLR']]).ravel()
    df['cci_scaled']   = cci_scaler.fit_transform(df[['CCI']]).ravel()
    return df


def train_hmm(X: np.ndarray, n_components: int = 2, random_state: int = 42) -> hmm.GaussianHMM:
    """Train a GaussianHMM with interpretable first dimension (raw log return)."""
    model = hmm.GaussianHMM(
        algorithm='map',
        n_components=int(n_components),
        covariance_type='full',
        n_iter=300,
        tol=1e-2,
        init_params='stmc',
        random_state=int(random_state)
    )
    model.fit(X)
    return model


def map_bull_bear_by_mean_logreturn(model: hmm.GaussianHMM) -> int:
    """Return the state index considered 'bull' (max mean on first feature)."""
    means = model.means_  # shape: (n_components, n_features)
    bull_state = int(np.argmax(means[:, 0]))
    return bull_state


def backtest_long_flat(close: pd.Series,
                       states: np.ndarray,
                       bull_state: int,
                       commission_factor: float = 1 - 0.04/100,
                       initial_cash: float = 100.0,
                       verbose: bool = False) -> float:
    """
    Simple long/flat strategy:
      - Long when state == bull_state, else flat.
      - Executes at next bar's close (proxy).
    Returns ratio vs. Buy & Hold.
    """
    # Alignment checks
    assert isinstance(close, pd.Series)
    assert len(close) == len(states)

    dates = close.index
    first_date = dates[0]
    last_date  = dates[-1]

    cash = float(initial_cash)
    shares = 0.0
    in_pos = False
    operations = 0

    # Buy & Hold benchmark
    bh_shares = initial_cash / close.loc[first_date]
    bh_final = bh_shares * close.loc[last_date]

    # Iterate through bars (execute on next bar)
    for i in range(len(dates) - 1):
        d_now = dates[i]
        d_next = dates[i + 1]
        go_long = (states[i] == bull_state)

        if go_long and (not in_pos):
            price = close.loc[d_next]
            shares = (cash * commission_factor) / price
            cash = 0.0
            in_pos = True
            operations += 1
        elif (not go_long) and in_pos:
            price = close.loc[d_next]
            cash = shares * price * commission_factor
            shares = 0.0
            in_pos = False
            operations += 1

    final_value = cash + shares * close.loc[last_date]
    ratio_vs_bh = final_value / bh_final

    if verbose:
        st.write(f'Operaciones: {operations}')
        st.write(f'B&H final: {bh_final:.2f} — Estrategia final: {final_value:.2f}')

    return ratio_vs_bh


def forecast_prices_by_means(model: hmm.GaussianHMM,
                             last_close: float,
                             current_state: int,
                             horizon: int = 15) -> pd.Series:
    """
    Deterministic forecast using argmax transition path:
      - At each step pick next state = argmax row of transition matrix.
      - Use mean of first feature (log-return) to propagate prices.
    """
    transition = model.transmat_
    means = model.means_

    state = int(current_state)
    preds = []
    for _ in range(int(horizon)):
        nxt = int(np.argmax(transition[state]))
        preds.append(means[nxt, 0])  # expected log-return for that state
        state = nxt

    prices = [float(last_close)]
    for r in preds:
        prices.append(prices[-1] * np.exp(r))

    # Build a business-day index forward from last date (to be assigned by caller)
    prices = prices[1:]  # drop seed
    return pd.Series(prices)


# -----------------------------
# Inputs
# -----------------------------
ticker = st.text_input('Especie', 'QQQ')
sma_period = int(st.number_input('Período de SMA', value=44, min_value=1, step=1))
ema_period = int(st.number_input('Período de EMA', value=30, min_value=1, step=1))
wma_period = int(st.number_input('Período de WMA', value=52, min_value=1, step=1))
rsi_period = int(st.number_input('Período de RSI', value=14, min_value=1, step=1))
willr_period = int(st.number_input('Período de WILLR', value=14, min_value=1, step=1))
cci_period = int(st.number_input('Período de CCI', value=20, min_value=1, step=1))
hmm_states = int(st.number_input('Número de estados HMM', value=2, min_value=2, max_value=8, step=1))
forecast_horizon = int(st.number_input('Horizonte de predicción (días hábiles)', value=15, min_value=1, step=1))

# -----------------------------
# Data download
# -----------------------------
data = yf.download(ticker, start="1970-01-01", end="2025-05-03", auto_adjust=False)
st.write(f'Longitud de los datos descargados: {len(data)}')

if data.empty:
    st.error("No se descargaron datos. Verifica el ticker o el rango de fechas.")
    st.stop()

# Ensure datetime index is sorted ascending
data = data.sort_index()

# -----------------------------
# Indicators and features
# -----------------------------
data = compute_indicators(
    df=data.copy(),
    sma_period=sma_period,
    ema_period=ema_period,
    wma_period=wma_period,
    rsi_period=rsi_period,
    willr_period=willr_period,
    cci_period=cci_period
)
data = scale_features(data)

# Feature matrix for HMM
feature_cols = ['Log Return', 'sma_scaled', 'cci_scaled', 'willr_scaled', 'ema_scaled', 'Range']
X = data[feature_cols].to_numpy()

# -----------------------------
# Train HMM
# -----------------------------
model = train_hmm(X, n_components=hmm_states, random_state=42)
hidden_states = model.predict(X)
log_likelihood = model.score(X)

# Report convergence
st.write(f'Converge: {model.monitor_.converged} — Iteraciones: {model.monitor_.iter}')
st.write(f'Log Likelihood: {log_likelihood:.0f}')

# Determine bull state by mean log-return
bull_state = map_bull_bear_by_mean_logreturn(model)
means_first_dim = model.means_[:, 0]
st.write(f'Estado bull (por media de log-return): {bull_state + 1}')
for i in range(model.n_components):
    st.write(f"μ_logret(Estado {i+1}): {means_first_dim[i]:+.6f}")

# -----------------------------
# Backtest (long/flat)
# -----------------------------
st.write('* BackTest *')
ratio_vs_bh = backtest_long_flat(
    close=data['Close'],
    states=hidden_states,
    bull_state=bull_state,
    commission_factor=(1 - 0.04/100),
    initial_cash=100.0,
    verbose=True
)
st.write(f'HMM/B&H: {ratio_vs_bh:.2f}')

# -----------------------------
# Forecast
# -----------------------------
last_date = data.index[-1]
last_close = data.loc[last_date, 'OHL']
current_state = int(hidden_states[-1])
st.write(f'Último estado observado: {current_state + 1} (fecha {last_date.date()})')

pred_series = forecast_prices_by_means(
    model=model,
    last_close=float(last_close),
    current_state=current_state,
    horizon=forecast_horizon
)
pred_index = pd.date_range(start=last_date + pd.DateOffset(days=1),
                           periods=len(pred_series),
                           freq='B')
predicted_df = pd.DataFrame({'OHL': pred_series.values}, index=pred_index)

# -----------------------------
# Plot (tail + forecast)
# -----------------------------
colors = ['red', 'green', 'blue', 'purple', 'orange', 'cyan', 'magenta', 'yellow']
fig = go.Figure()

tail = min(6000, len(data))
tail_df = data.tail(tail)
states_tail = hidden_states[-tail:]

for i in range(model.n_components):
    mask = (states_tail == i)
    x = tail_df.index[mask]
    y = tail_df['OHL'].to_numpy()[mask]
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
    title=f'Precio de cierre de {ticker}: estados ocultos y proyección (HMM)',
    xaxis_title='Fecha',
    yaxis_title='Precio de cierre (log)',
    yaxis_type='log',
    autosize=True,
    margin=dict(l=20, r=20, t=40, b=20),
    template="plotly_dark"
)

st.plotly_chart(fig, use_container_width=True)
