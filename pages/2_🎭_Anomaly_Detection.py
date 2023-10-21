import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import streamlit as st

st.markdown("<h1 style='text-align: center;'>Anomaly Detection</h1>", unsafe_allow_html=True)
df = st.file_uploader("Upload your file here...")


if df is not None:  # Check if a file is uploaded
    # Read the CSV data only if a file is uploaded
    log_data = pd.read_csv(df)

    log_data['Time'] = pd.to_datetime(log_data['Time'], format='%d%m%Y', errors='ignore')

    log_data['URL'] = log_data['URL'].map(lambda x: x.lstrip('0'))

    log_data['month'] = log_data['Time'].str.slice(3, 6)
    log_data['day'] = log_data['Time'].str.slice(0, 2)

    log_data['day'] = pd.to_numeric(log_data['day'], errors='coerce')  # Convert non-numeric values to NaN
    log_data['day'] = log_data['day'].clip(lower=1, upper=30)

    log_data['Methods'] = log_data['URL'].str.split('/').str[0]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<h1 style='text-align: center;'>Method 1</h1>", unsafe_allow_html=True)
        # Select relevant features for anomaly detection
        selected_features = ['IP', 'URL', 'Status']

        # Filter the DataFrame to remove rows with '2018]' and '2017]' in 'Status' column
        df_filtered = log_data[~log_data['Status'].str.contains('2018\]')]
        df_filtered = df_filtered[~df_filtered['Status'].str.contains('2017\]')]
        #print(df_filtered["Status"].value_counts())

        # Initialize label encoder
        label_encoder = LabelEncoder()

        # Encode the 'Status' column in the filtered DataFrame
        df_filtered['Status_encoded'] = label_encoder.fit_transform(df_filtered['Status'])

        # Encode categorical features (IP and URL) to numeric values
        data_encoded = pd.get_dummies(df_filtered[selected_features], drop_first=True)

        #print(data_encoded.head(5))

        # Fit an Isolation Forest model
        model = IsolationForest(contamination=0.05, random_state=42)  # Adjust contamination as needed
        model.fit(data_encoded)

        # Predict anomalies (-1 for anomalies, 1 for normal data)
        anomaly_predictions = model.predict(data_encoded)

        # Add the anomaly predictions to the original filtered dataset
        df_filtered['Anomaly'] = anomaly_predictions

        # Filter and display anomalies
        anomalies = df_filtered[df_filtered['Anomaly'] == -1]  # -1 indicates an anomaly
        prettyfied_anomalies = anomalies.iloc[:, [0, 1, 2, 3]]
        print(anomalies)
        st.write("Detected Anomalies:")
        st.write(prettyfied_anomalies)
        #["IP", "Time", "URL", "Status", "Anomaly"].head())

    with col2:
        from scipy.stats import zscore
        st.markdown("<h1 style='text-align: center;'>Method 2</h1>", unsafe_allow_html=True)
        threshold = st.slider("Threshold for Z-Score Anomaly Detection", min_value=1.0, max_value=1.9, value=1.5,
                              step=0.05)
        st.write("Best above 1.6 threshold")


        def detect_anomalies(data, threshold):
            grouped = data.groupby(['Time', 'IP', 'Status']).size().reset_index(name='TrafficCount')
            grouped['ZScore'] = grouped.groupby(['Time', 'Status'])['TrafficCount'].transform(zscore)
            anomalies = grouped[grouped['ZScore'] > threshold]
            return anomalies


        def visualize_detected_anomalies(anomalies):
            st.subheader("Detected Anomalies:")
            st.write(anomalies)
            # Visualization code here


        def visualize_anomalies_per_ip(anomaly_counts):
            st.subheader("Anomalies Detected per IP")
            st.bar_chart(anomaly_counts.set_index('IP'))


        if st.button("Detect Anomalies"):
            threshold_slider_val = threshold  # Set your desired threshold
            anomalies = detect_anomalies(log_data, threshold)
            visualize_detected_anomalies(anomalies)

            # Count anomalies per IP
            anomaly_counts = anomalies.groupby('IP').size().reset_index(name='AnomalyCount')
            visualize_anomalies_per_ip(anomaly_counts)

            # Find IP with the highest TrafficCount
            ip_with_most_traffic = anomaly_counts.loc[anomaly_counts['AnomalyCount'].idxmax(), 'IP']
            st.subheader(f"IP with the Most Traffic: [{ip_with_most_traffic}] within the given threshold")
