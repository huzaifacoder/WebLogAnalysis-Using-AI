import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from main import *

# Create a Streamlit app
st.title("ARIMA Time Series Forecasting")

st.markdown("### Sample Data")
st.write(data.head())

# Visualize the data
st.markdown("### Time Series Data")
plt.figure(figsize=(12, 6))
plt.plot(data['Time'], data['Status'])
plt.title('Time Series Data')
plt.xlabel('Time')
plt.ylabel('Status')
st.pyplot()


# Choosing Parameters
st.markdown("### Autocorrelation and Partial Autocorrelation")
plt.figure(figsize=(12, 6))
plot_acf(data, lags=50)
plot_pacf(data, lags=50)
st.pyplot()

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
plt.figure(figsize=(12, 6))
plt.plot(data['day'], data['Status'], label='Observed')
plt.plot(pd.date_range(start=data['Time'].iloc[-1], periods=forecast_steps + 1, freq='D')[1:], forecast, label='Forecast', color='orange')
plt.fill_between(pd.date_range(start=data['Time'].iloc[-1], periods=forecast_steps + 1, freq='D')[1:], forecast - stderr, forecast + stderr, color='orange', alpha=0.3)
plt.title('ARIMA Forecast')
plt.xlabel('Time')
plt.ylabel('Status')
plt.legend()
st.pyplot()
