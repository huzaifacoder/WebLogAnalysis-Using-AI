import os
import pandas as pd

# Month abbreviation to number mapping dictionary
month_mapping = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
}

# Load the data into a DataFrame (replace this with your actual data loading method)
# data = pd.read_csv("your_data.csv")
# Load the data into a DataFrame (replace this with your actual data loading method)
cwd_join = os.getcwd() + "\\"
database_rel = os.path.relpath('Database\\weblog_cleaned.csv')
database_abs = cwd_join + database_rel
data = pd.read_csv(database_abs)

# Remove "[" character from the 'Time' column
data['Time'] = data['Time'].str.replace('[', '')

datetime = pd.to_datetime(data['Time'], format='%d/%b/%Y:%H:%M:%S')
data['Time_cleaned'] = datetime
data.set_index('Time', inplace=True)

print(datetime)

datetime.plot()

from statsmodels.tsa.stattools import adfuller

# Perform Augmented Dickey-Fuller test
result = adfuller(data['Status'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])
print('d-value:', result[2])
print('q-value:', result[3])

if result[1] > 0.05:
    # Apply differencing to make the series stationary
    data['Differenced_Status'] = data['Status'].diff().dropna()

    # Perform Augmented Dickey-Fuller test on the differenced series
    result_diff = adfuller(data['Differenced_Status'])
    print('ADF Statistic (Differenced):', result_diff[0])
    print('p-value (Differenced):', result_diff[1])
else:
    print('The original series is already stationary.')


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

# Plot ACF and PACF to determine the orders
plot_acf(data['Status'])
plot_pacf(data['Status'])
plt.show()


from statsmodels.tsa.statespace.sarimax import SARIMAX

# Fit ARIMA model
p, d, q = (1, 1, 2)  # Replace with actual values
model = SARIMAX(data['Status'], order=(p, d, q))
results = model.fit()

# Print model summary
print(results.summary())


# Forecast the next n steps
n = 2  # Replace with the number of steps you want to forecast

forecast, conf_int = results.forecast(steps=n, alpha=0.05)
print('Forecast:', forecast)
print('Confidence Interval:', conf_int)


import matplotlib.pyplot as plt
import numpy as np


# Convert the datetime Series to a pandas datetime format (assuming datetime is your datetime Series)
datetime = np.array(datetime)

# Convert the 'Status' column to a numpy array
status_numpy = np.array(data['Status'])

# Get the first datetime value to use as the start for date_range
start_datetime = datetime.iloc[0]

# Calculate the length of the datetime Series
num_periods = len(datetime)

# Create the forecast_index using date_range
forecast_index = pd.date_range(start=start_datetime, periods=num_periods, freq='D')

# Plot the original series, forecast, and confidence intervals
plt.figure(figsize=(10, 6))
plt.plot(datetime, status_numpy, label='Original Data', color='blue')
plt.plot(forecast_index, forecast, label='Forecast', color='red')
plt.fill_between(forecast_index, conf_int[:, 0], conf_int[:, 1], color='pink', alpha=0.3, label='Confidence Intervals')
plt.title('ARIMA Forecast')
plt.xlabel('Time')
plt.ylabel('Status')
plt.legend()
plt.show()
