import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import streamlit as st
st.title("Anomaly Detection")
# Load the sample dataset
df = pd.read_csv("Database\\weblog.csv")

# Select relevant features for anomaly detection
selected_features = ['IP', 'URL', 'Status']

# Filter the DataFrame to remove rows with '2018]' and '2017]' in 'Status' column
df_filtered = df[~df['Status'].str.contains('2018\]')]
df_filtered = df_filtered[~df_filtered['Status'].str.contains('2017\]')]
print(df_filtered["Status"].value_counts())

# Initialize label encoder
label_encoder = LabelEncoder()

# Encode the 'Status' column in the filtered DataFrame
df_filtered['Status_encoded'] = label_encoder.fit_transform(df_filtered['Status'])

# Encode categorical features (IP and URL) to numeric values
data_encoded = pd.get_dummies(df_filtered[selected_features], drop_first=True)

print(data_encoded.head(5))

# Fit an Isolation Forest model
model = IsolationForest(contamination=0.05, random_state=42)  # Adjust contamination as needed
model.fit(data_encoded)

# Predict anomalies (-1 for anomalies, 1 for normal data)
anomaly_predictions = model.predict(data_encoded)

# Add the anomaly predictions to the original filtered dataset
df_filtered['Anomaly'] = anomaly_predictions

# Filter and display anomalies
anomalies = df_filtered[df_filtered['Anomaly'] == -1]  # -1 indicates an anomaly
print("Detected Anomalies:")
print(anomalies)