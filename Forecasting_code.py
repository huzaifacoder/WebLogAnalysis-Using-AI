import pandas as pd
import statsmodels
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import streamlit as st
import time
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os

log_data = pd.read_csv("weblog.csv")

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
unique_ips = log_data['IP'].nunique()  # Count the number of unique IP addresses
unique_urls = log_data['URL'].nunique()  # Count the number of unique URLs
status_counts = log_data['Status'].value_counts()  # Count the occurrences of each status code
# Perform exploratory analysis
total_requests = len(log_data)  # Total number of requests
unique_visitors = log_data['IP'].nunique()  # Number of unique IP addresses
aggregated_data = log_data.groupby('Date').agg({'IP': 'count'}).reset_index()
aggregated_data.columns = ['Date', 'request_count']
aggregated_data.set_index('Date', inplace=True)
target_variable = aggregated_data['request_count']
target_variable.plot(figsize=(12, 6))
plt.xlabel('Timestamp')
plt.ylabel('Total Requests')
plt.title('Web Log Data')
plt.show()
# Normalize data
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(target_variable.values.reshape(-1, 1))

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
pred_no = 10
seq_length = 10  # Choose an appropriate sequence length
train_sequences = create_sequences(train_data, seq_length)
test_sequences = create_sequences(test_data, seq_length)

# Convert sequences to numpy arrays
X_train, y_train = np.array([seq for seq, target in train_sequences]), np.array([target for seq, target in train_sequences])
X_test, y_test = np.array([seq for seq, target in test_sequences]), np.array([target for seq, target in test_sequences])

# Build RNN model
model = Sequential([
    LSTM(64, activation='relu', input_shape=(seq_length, 1)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=64, verbose=1)

# Make predictions
predictions = model.predict(X_test)

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
