import pandas as pd
import statsmodels
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import time
import os

cwd_join = os.getcwd() + "\\"
database_rel = os.path.relpath('Database\\weblog_cleaned.csv')
database_abs = cwd_join + database_rel
log_data = pd.read_csv(database_abs)

log_data = pd.read_csv('Database\\weblog_cleaned.csv')
print(log_data.head())

# Create new columns for day, month, year, and time
log_data['Date'] = log_data['Time'].str.extract(r'\[(\d{2}/\w+/\d{4})')
log_data['Day'] = log_data['Date'].str.extract(r'(\d{2})/')
log_data['Month'] = log_data['Date'].str.extract(r'/(\w+)/')
log_data['Year'] = log_data['Date'].str.extract(r'/(\d{4})')
log_data['Time'] = log_data['Time'].str.extract(r':(\d{2}:\d{2}:\d{2})')
log_data['URL'] = log_data['URL'].str.extract(r'(\S+)\sHTTP/1\.1')

page_views = log_data['URL'].value_counts()  # Count of page views for each URL


# Perform analysis based on available columns
unique_ips = log_data['IP'].unique()  # Count the number of unique IP addresses

unique_urls = log_data['URL'].unique()  # Count the number of unique URLs
status_counts = log_data['Status'].value_counts()  # Count the occurrences of each status code


#print(unique_ips)
#print(unique_ips)
#print(unique_ips)


# Perform exploratory analysis
total_requests = len(log_data["IP"])  # Total number of requests
print(total_requests)
print(total_requests)
print(total_requests)

unique_visitors = log_data['IP'].nunique()  # Number of unique IP addresses
aggregated_data = log_data.groupby('Date').agg({'IP': 'count'}).reset_index()


aggregated_data.columns = ['Date', 'request_count']
aggregated_data.set_index('Date', inplace=True)

target_variable = aggregated_data['request_count']
print()
print(aggregated_data['request_count'].head(15))
print()

import re
df_str = aggregated_data['request_count'].to_string(index=False)

pattern = r"\S+\s+(\d+)"

# Find all matches using the regex pattern
matches = re.findall(pattern, df_str)

# Store the extracted data in a list
extracted_data = [int(match) for match in matches]

np.array(extracted_data)
print(extracted_data)


target_variable.plot(figsize=(12, 6))
plt.xlabel('Timestamp')
plt.ylabel('Total Requests')
plt.title('Web Log Data')
plt.show()

# Step 3: Make the Time Series Stationary
# Apply differencing to remove trend and seasonality
differenced_data = target_variable.diff().dropna()

# Step 4: Determining ARIMA Parameters (For DEV use ONLY)
# Plot the autocorrelation and partial autocorrelation functions


#fig, ax = plt.subplots(figsize=(12, 6))
#plot_acf(differenced_data, ax=ax, lags=30)
#plt.xlabel('Lags')
#plt.ylabel('Autocorrelation')
#plt.title('Autocorrelation Function')
#plt.show()


#fig, ax = plt.subplots(figsize=(12, 6))
#plot_pacf(differenced_data, ax=ax, lags=25)
#plt.xlabel('Lags')
#plt.ylabel('Partial Autocorrelation')
#plt.title('Partial Autocorrelation Function')
#plt.show()


# Determine the order (p, d, q) of the ARIMA model based on the plots and ADF test

# Step 5: Split the Data
train_data = differenced_data[:-30]  # Use all but the last 30 data points for training
test_data = differenced_data[-30:]  # Use the last 30 data points for testing

# Convert DataFrame or Series to NumPy array before indexing
train_data = train_data.to_numpy()
train_data = train_data[:, None]  # Perform the desired indexing on the NumPy array

# Convert DataFrame or Series to NumPy array before indexing
test_data = test_data.to_numpy()
test_data = test_data[:, None]  # Perform the desired indexing on the NumPy array


# Step 6: Fit the ARIMA Model

order = (2, 3, 1)
model = ARIMA(train_data, order=order)
model_fit = model.fit()

# Step 8: Evaluate the Model
predictions = model_fit.forecast(steps=len(test_data))
mse = mean_squared_error(test_data, predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(test_data, predictions)
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")

pred_no = 10

future_predictions = model_fit.forecast(steps=int(pred_no))  # Example: Generate 10 future predictions

print(future_predictions)  # Example: Generate 10 future predictions


import matplotlib.pyplot as plt

# Assuming denormalized_future_predictions contains your predicted values
days_ahead = list(range(1, pred_no + 1))  # Adjust the range based on the number of days predicted

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(days_ahead, future_predictions, marker='o', linestyle='-', color='b', label='Predicted Values')
plt.xlabel('Days Ahead')
plt.ylabel('Predicted Requests')
plt.title('Predicted Values for Future Days')
plt.legend()
plt.grid()
plt.show()



######################################



import pandas as pd
import statsmodels
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
#import plotly.graph_objects as go
#import plotly.express as px
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import streamlit as st
import plotly.graph_objects as go
import time
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation, Dropout


# Normalize data
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(target_variable.values.reshape(-1, 1))
print(normalized_data.shape)

# Split data into train and test sets
train_size = int(len(normalized_data) * 0.8)
train_data, test_data = normalized_data[:train_size], normalized_data[train_size:]


# Create sequences for training
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        target = data[i + seq_length]
        sequences.append((seq, target))
    return sequences

seq_length = 10  # Choose an appropriate sequence length
train_sequences = create_sequences(train_data, seq_length)
test_sequences = create_sequences(test_data, seq_length)

# Convert sequences to numpy arrays
X_train, y_train = np.array([seq for seq, target in train_sequences]), np.array([target for seq, target in train_sequences])
X_test, y_test = np.array([seq for seq, target in test_sequences]), np.array([target for seq, target in test_sequences])

# Build RNN model
model = Sequential([
    LSTM(128, activation='relu', input_shape=(seq_length, 1), return_sequences=True),
    Dropout(0.2),  # Add dropout for regularization
    LSTM(128, activation='relu', return_sequences=True),
    Dropout(0.2),
    LSTM(64, activation='relu', return_sequences=True),
    Dropout(0.2),
    LSTM(32, activation='relu', return_sequences=True),
    Dropout(0.2),
    LSTM(64, activation='relu'),
    Dense(64),
    Activation('relu'),
    Dropout(0.2),
    Dense(32),
    Activation('relu'),
    Dense(1, activation='linear')  # Use linear activation for regression
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

# Train the model
model.fit(X_train, y_train, epochs=45, batch_size=32, verbose=1)

# Make predictions
predictions = model.predict(X_test) #############################################################

# Denormalize predictions
predictions = scaler.inverse_transform(predictions)

# Evaluate the model
mse = mean_squared_error(test_data[seq_length:], predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(test_data[seq_length:], predictions)
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")

# Generate future predictions
future_input = normalized_data[-seq_length:].reshape(1, seq_length, 1)
future_predictions_normalized = model.predict(future_input)
future_predictions = scaler.inverse_transform(future_predictions_normalized)

print(future_predictions)


#To generate prediction for multiple days, you need to use rolling forecast as well

# Generate future predictions
future_input = normalized_data[-seq_length:].reshape(1, seq_length, 1)
future_predictions = []

for _ in range(pred_no):  # Replace pred_no with the number of days you want to predict
    future_prediction_normalized = model.predict(future_input)
    future_predictions.append(future_prediction_normalized[0, 0])
    future_input = np.append(future_input[:, 1:, :], future_prediction_normalized.reshape(1, 1, 1), axis=1)

# Denormalize future predictions
denormalized_future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

print(denormalized_future_predictions)


import matplotlib.pyplot as plt

# Assuming denormalized_future_predictions contains your predicted values
days_ahead = list(range(1, pred_no + 1))  # Adjust the range based on the number of days predicted

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(days_ahead, denormalized_future_predictions, marker='o', linestyle='-', color='b', label='Predicted Values')
plt.xlabel('Days Ahead')
plt.ylabel('Predicted Value')
plt.title('Predicted Values for Future Days')
plt.legend()
plt.grid()
plt.show()

max_value = 5000
min_value = 100


# Denormalize the predictions
denormalized_predictions = denormalized_future_predictions * (max_value - min_value) + min_value
#denormalized_value = 0.5 * (normalized_data + 1) * (max_value - min_value) + min_value


# Plotting both original data and denormalized forecasted predictions
plt.figure(figsize=(12, 6))

# Plot denormalized forecasted predictions in blue
plt.plot(days_ahead, denormalized_predictions, marker='o', linestyle='-', color='blue', label='Predicted Values')

plt.xlabel('years ahead')
plt.ylabel('Total Requests')
plt.title('Web Log Data and Denormalized Predicted Values')
plt.legend()
plt.grid()

plt.show()
