import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
from keras.optimizers import Adam
from keras.losses import MeanSquaredError

# Load your time series data
# Assuming you have a single time series column named 'Value'
data = pd.read_csv("Database\\weblog.csv")

data['Time'] = data['Time'].str.lstrip('[')
data[['Date', 'Time']] = data['Time'].str.split(':', n=1, expand=True)

data['month'] = data['Time'].str.slice(3, 6)
data['day'] = data['Time'].str.slice(0, 2)

data['day'] = pd.to_numeric(data['day'], errors='coerce')  # Convert non-numeric values to NaN
data['day'] = data['day'].clip(lower=1, upper=30)

values = data['day'].values.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler()
scaled_values = scaler.fit_transform(values)

# Split data into training and testing sets
train_size = int(len(scaled_values) * 0.8)
train_data, test_data = scaled_values[:train_size], scaled_values[train_size:]

# Create sequences for LSTM training
def create_sequences(data, sequence_length):
    X = []
    y = []
    for i in range(len(data) - sequence_length):
        seq = data[i:i + sequence_length]
        label = data[i + sequence_length]
        X.append(seq)
        y.append(label)
    return np.array(X), np.array(y)

sequence_length = 10  # Adjust as needed
X_train, y_train = create_sequences(train_data, sequence_length)
X_test, y_test = create_sequences(test_data, sequence_length)

# Reshape X_train and X_test for LSTM input (samples, time steps, features)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Create features: IP, Time, Status
data['IP'] = data['IP']
data['TimeInterval'] = data['Time']  # You can adjust the time interval granularity
data['Status'] = data['Status']

# Prepare features data for X_train and X_test
X_train_features = data.iloc[:train_size, ['IP', 'TimeInterval', 'Status']].values
X_test_features = data.iloc[train_size - sequence_length:-sequence_length, ['IP', 'TimeInterval', 'Status']].values

# Functional API: Create LSTM input for features
input_features = Input(shape=(3,))  # Assuming you have 3 features: IP, TimeInterval, Status
lstm_input = Input(shape=input_shape)
lstm_out = LSTM(50, activation='relu')(lstm_input)
merged = keras.layers.concatenate([lstm_out, input_features])
output = Dense(1)(merged)

# Define the model
model = Model(inputs=[lstm_input, input_features], outputs=output)
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit([X_train, X_train_features], y_train, epochs=50, batch_size=32,
          validation_data=([X_test, X_test_features], y_test))

# Forecast using the trained model
forecast = model.predict([X_test, X_test_features])
forecast = scaler.inverse_transform(forecast)

# Plot the original data and the forecast
plt.plot(np.arange(len(values)), values, label='Original Data')
plt.plot(np.arange(train_size + sequence_length, len(values)), forecast, label='Forecast', linestyle='dashed')
plt.legend()
plt.show()