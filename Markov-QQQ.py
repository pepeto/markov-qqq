import streamlit as st
import ta
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV, train_test_split

# Configurar el diseño de la página
st.set_page_config(layout="wide")

st.title('Análisis de Estados Ocultos con HMM')

especie = st.text_input('Especie', 'QQQ')
sma_period = st.number_input('Período de SMA', value=44, min_value=1, step=1)
ema_period = st.number_input('Período de EMA', value=30, min_value=1, step=1)
wma_period = st.number_input('Período de WMA', value=52, min_value=1, step=1)
rsi_period = st.number_input('Período de RSI', value=14, min_value=1, step=1)
willr_period = st.number_input('Período de WILLR', value=14, min_value=1, step=1)
cci_period = st.number_input('Período de CCI', value=20, min_value=1, step=1)

data = yf.download(especie, start="1970-01-01", end="2025-05-03")
st.write(f'Longitud de los datos: {len(data)}')

# TRANSFORMACION
scaler = StandardScaler()
MM_scaler = MinMaxScaler()
data['OHL'] = data['Close']
data['Log Return'] = np.log(data['OHL'] / data['OHL'].shift(1))
log_returns = data['Log Return'].values.reshape(-1, 1)

data['SMA'] = data['OHL'].rolling(window=sma_period).mean()
data['SMA_log_return'] = np.log(data['SMA'] / data['SMA'].shift(1))
data['sma_scaled'] = scaler.fit_transform(data['SMA_log_return'].values.reshape(-1, 1))

data['EMA'] = ta.trend.EMAIndicator(data['OHL'], window=ema_period).ema_indicator()
data['EMA_log_return'] = np.log(data['EMA'] / data['EMA'].shift(1))
data['ema_scaled'] = scaler.fit_transform(data['EMA_log_return'].values.reshape(-1, 1))

data['WMA'] = ta.trend.WMAIndicator(data['OHL'], window=wma_period).wma()
data['WMA_log_return'] = np.log(data['WMA'] / data['WMA'].shift(1))
data['wma_scaled'] = scaler.fit_transform(data['WMA_log_return'].values.reshape(-1, 1))

data['RSI'] = ta.momentum.RSIIndicator(data['OHL'], window=rsi_period).rsi()
data['rsi_scaled'] = MM_scaler.fit_transform(data['RSI'].values.reshape(-1, 1))

data['WILLR'] = ta.momentum.WilliamsRIndicator(data['High'], data['Low'], data['Close'], lbp=willr_period).williams_r()
data['willr_scaled'] = MM_scaler.fit_transform(data['WILLR'].values.reshape(-1, 1))

data['CCI'] = ta.trend.CCIIndicator(data['High'], data['Low'], data['Close'], window=cci_period).cci()
data['cci_scaled'] = MM_scaler.fit_transform(data['CCI'].values.reshape(-1, 1))

data['Range'] = (data['High'] / data['Low']) - 1
data.dropna(inplace=True)

def BackTest(hidden_states, inversion=False, verbose=0, comision=1 - 0.04/100, capital_inicial=100):
    cash = capital_inicial
    acciones = 0
    operaciones = 0
    comprado = False

    # Buy and Hold
    acciones_BH = capital_inicial / data['Close'].iloc[0]
    valor_final_BH = acciones_BH * data['Close'].iloc[-1]

    for i in range(len(data) - 1):
        if (inversion ^ (hidden_states[i] == 1)) and not comprado:
            acciones = cash * comision / data['Close'].iloc[i + 1]
            cash = 0
            comprado = True
            operaciones += 1
        elif (inversion ^ (hidden_states[i] == 0)) and comprado:
            cash = acciones * data['Close'].iloc[i + 1] * comision
            acciones = 0
            comprado = False
            operaciones += 1

    if verbose == 1:
        st.write(f'B&H: {valor_final_BH:.2f}')
        st.write(f'Operaciones: {operaciones} - Op/día: {operaciones / 60 / 24:.2f}')

    return (cash + acciones * valor_final_BH) / valor_final_BH

features_scaled = data[['Log Return', 'sma_scaled', 'cci_scaled', 'willr_scaled', 'ema_scaled', 'Range']]
model = hmm.GaussianHMM(algorithm='map', n_components=2, covariance_type='full', n_iter=300, tol=0.01, init_params='stmc')
model.fit(features_scaled)

hidden_states = model.predict(features_scaled)
log_likelihood = model.score(features_scaled)
st.write(f'Converge: {model.monitor_.converged} - Iter.: {model.monitor_.iter}')
st.write(f'Log Likelihood: {log_likelihood:.0f}')

st.write('* BackTest *')
BackTest(hidden_states, False, 1)
st.write(f'HMM/B&H: {max(BackTest(hidden_states, True), BackTest(hidden_states, False)):.2f}')

# Preparar las predicciones más probables
transition_probs = model.transmat_
emission_means = model.means_
emission_covars = model.covars_

current_state = hidden_states[-1]
predicted_states = [current_state]

for _ in range(15):
    next_state = np.argmax(transition_probs[current_state])
    predicted_states.append(next_state)
    current_state = next_state

predicted_observations = [emission_means[state][0] for state in predicted_states[1:]]
predicted_returns = scaler.inverse_transform(np.array(predicted_observations).reshape(-1, 1))

last_close_price = data['OHL'].iloc[-1]
predicted_close_prices = [last_close_price]

for ret in predicted_returns:
    predicted_close_prices.append(predicted_close_prices[-1] * np.exp(ret[0]))

last_date = data.index[-1]
predicted_dates = pd.date_range(start=last_date + pd.DateOffset(days=1), periods=len(predicted_returns), freq='B')
predicted_data = pd.DataFrame({'OHL': predicted_close_prices[1:]}, index=predicted_dates)

# Definir colores para los estados ocultos para la visualización
colors = ['red', 'green', 'blue', 'purple', 'orange']

# Crear un objeto de figura de Plotly
fig = go.Figure()

# Añadir trazas para cada estado oculto con diferentes colores
for i in range(model.n_components):
    mask = hidden_states[-6000:] == i
    fig.add_trace(go.Scatter(
        x=data.index[-6000:][mask],
        y=data['OHL'][-6000:][mask],
        mode='markers',
        opacity=0.6,
        marker=dict(color=colors[i % len(colors)], size=2),
        name=f'Estado {i + 1}'
    ))
	
# Añadir traza para los precios de cierre predichos
fig.add_trace(go.Scatter(
    x=predicted_data.index,
    y=predicted_data['OHL'],
    mode='lines+markers',
    name='Pred.',
    line=dict(color='grey')
))

# Actualizar el diseño del gráfico
fig.update_layout(
    title=f'Precio de cierre del {especie} estados ocultos y predicción más probable',
    xaxis_title='Fecha',
    yaxis_title='Precio de cierre',
    yaxis_type='log',
    autosize=True,
    margin=dict(l=20, r=20, t=40, b=20),
    template="plotly_dark"
)

# Mostrar el gráfico en un contenedor más ancho
st.plotly_chart(fig, use_container_width=True, width=1000, height=800)
