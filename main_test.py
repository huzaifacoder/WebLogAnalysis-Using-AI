import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from main import data

# Restrict the 'day' column values to the range of 1 to 30
data['day'] = pd.to_numeric(data['day'], errors='coerce')  # Convert non-numeric values to NaN
data['day'] = data['day'].clip(lower=1, upper=30)

# Create a Streamlit app
st.title("ARIMA Time Series Forecasting")

st.markdown("### Sample Data")
st.write(data.head())

# Visualize the data
st.markdown("### Time Series Data")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(data['day'], data['Status'])
ax.set_title('Time Series Data')
ax.set_xlabel('Day')
ax.set_ylabel('Status')
st.pyplot(fig)


# Choosing Parameters
st.markdown("### Autocorrelation and Partial Autocorrelation")
fig, ax = plt.subplots(figsize=(12, 6))
plot_acf(data["day"], lags=50, ax=ax)
plot_pacf(data["day"], lags=50, ax=ax)
st.pyplot(fig)

# Build and Train ARIMA Model
p = 1  # AR order
d = 1  # Differencing order
q = 1  # MA order

model = ARIMA(data['Status'], order=(p, d, q))
model_fit = model.fit(disp=0)

# Forecasting
forecast_steps = 30  # Number of steps to forecast
forecast, stderr, conf_int = model_fit.forecast(steps=forecast_steps)

# Visualize Results
st.markdown("### ARIMA Forecast")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(data['Time'], data['Status'], label='Observed')
ax.plot(pd.date_range(start=data['Time'].iloc[-1], periods=forecast_steps + 1, freq='D')[1:], forecast, label='Forecast', color='orange')
ax.fill_between(pd.date_range(start=data['Time'].iloc[-1], periods=forecast_steps + 1, freq='D')[1:], forecast - stderr, forecast + stderr, color='orange', alpha=0.3)
ax.set_title('ARIMA Forecast')
ax.set_xlabel('Time')
ax.set_ylabel('Status')
ax.legend()
st.pyplot(fig)
