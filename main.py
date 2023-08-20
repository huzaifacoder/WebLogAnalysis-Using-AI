import pandas as pd
import os
from datetime import datetime

cwd_join = os.getcwd() + "\\"
database_rel = os.path.relpath('Database\\weblog.csv')
database_abs = cwd_join + database_rel
data = pd.read_csv(database_abs)




data['Time'] = pd.to_datetime(data['Time'], format='%d%m%Y', errors='ignore')
#print(data['Time'].head())

#data = data.rename(columns={'Staus': 'Status'}, index={'ONE': 'one'})
data['URL'] = data['URL'].map(lambda x: x.lstrip('0'))
data.describe()

data['month'] = data['Time'].str.slice(3, 6)
data['day'] = data['Time'].str.slice(0, 2)

data['day'] = pd.to_numeric(data['day'], errors='coerce')  # Convert non-numeric values to NaN
data['day'] = data['day'].clip(lower=1, upper=30)

data['Methods'] = data['URL'].str.split('/').str[0]

# if data['URL'].str.contains('.js').any():
#   data['URL_new'] = data['URL'].str.split('/').str[3]
if data['URL'].str.contains('.php').any():
    data['URL_new'] = data['URL'].str.split('/').str[1]
elif data['URL'].str.contains('.js').any():
    data['URL_new'] = data['URL'].str.split('/').str[3]


# Detecting Malicious Activity
malicious_extension = 'php?id='  # Example malicious extensions


# Define a function to check for malicious activity
def is_malicious(x):
    if isinstance(x, str):  # Check if x is a string
        return any(ext in x for ext in malicious_extension)
    return False


data['Malicious'] = data['URL_new'].apply(is_malicious)

# Count the occurrences of malicious URLs
malicious_counts = data[data['Malicious']].groupby('URL_new').size().reset_index(name='Count')

# Display URLs with high occurrence counts
print("Malicious URLs with high occurrence counts:")
print(malicious_counts.sort_values(by='Count', ascending=False).head(10))

print(data['day'].unique())


#STREAMLIT
import streamlit as st

st.set_page_config(page_title="Web Log Analysis", page_icon=":desktop_computer:", layout = "wide")


# Function to display Status distribution plot
def display_status_distribution():
    st.markdown("#### Status Distribution")
    st.bar_chart(data['Status'].value_counts().head(40))


# Function to display Day-wise distribution plot
def display_daywise_distribution():
    st.markdown("#### Day-wise Distribution")
    st.bar_chart(data['day'].value_counts().head(40))


# Function to display Month-wise distribution plot
def display_monthwise_distribution():
    st.markdown("#### Month-wise Distribution")
    return st.bar_chart(data['month'].value_counts().head(40))


# Function to display HTTP Methods distribution plot
def display_methods_distribution():
    st.markdown("#### Methods  Distribution")
    return st.bar_chart(data['Methods'].value_counts().head(40))

select_pages_list = ["Status Distribution", "Day-wise Distribution", "Month-wise Distribution", "Methods Distribution"]

# Display selected page content
st.markdown("<h1 style='text-align: center; font-size: 100px;'>Web log Analysis \n</h1>", unsafe_allow_html=True)
st.title("*\n---")
print(data["Time"].head())

# Create features: IP, Time, Status
data['IP'] = data['IP']
data['TimeInterval'] = data['Time']  # You can adjust the time interval granularity
data['Status'] = data['Status']


from scipy.stats import zscore


# Right-side content (Traffic Anomaly Detection)
col1, col2 = st.columns(2)

with col1:
    st.markdown("<h1 style='text-align: center;'>Basic Statistics</h1>", unsafe_allow_html=True)

    # Sidebar navigation
    selected_page = st.selectbox("", select_pages_list)

    if selected_page == "Status Distribution":
        display_status_distribution()
    elif selected_page == "Day-wise Distribution":
        display_daywise_distribution()
    elif selected_page == "Month-wise Distribution":
        display_monthwise_distribution()
    elif selected_page == "Methods Distribution":
        display_methods_distribution()


with col2:
    st.markdown("<h1 style='text-align: center;'>Traffic Anomaly Detection</h1>", unsafe_allow_html=True)

    threshold = st.slider("Threshold for Z-Score Anomaly Detection", min_value=1.0, max_value=1.9, value=1.5, step=0.05)
    st.write("Best below 1.6 threshold")


    def detect_anomalies(data, threshold):
        grouped = data.groupby(['TimeInterval', 'IP', 'Status']).size().reset_index(name='TrafficCount')
        grouped['ZScore'] = grouped.groupby(['TimeInterval', 'Status'])['TrafficCount'].transform(zscore)
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
        anomalies = detect_anomalies(data, threshold)
        visualize_detected_anomalies(anomalies)

        # Count anomalies per IP
        anomaly_counts = anomalies.groupby('IP').size().reset_index(name='AnomalyCount')
        visualize_anomalies_per_ip(anomaly_counts)

        # Find IP with the highest TrafficCount
        ip_with_most_traffic = anomaly_counts.loc[anomaly_counts['AnomalyCount'].idxmax(), 'IP']
        st.subheader(f"IP with the Most Traffic: [{ip_with_most_traffic}] within the given threshold")


# Replace "/" with "-" and map months, and remove "["
def preprocess_time(time_str):
    if time_str.startswith('['):
        time_str = time_str.replace('/', '-').replace('[', '')
        datetime_obj = datetime.strptime(time_str, '%d-%b-%Y:%H:%M:%S')
        formatted_date = datetime_obj.strftime('%Y-%m-%d')
        month_numeric = datetime_obj.month
        return datetime_obj, formatted_date, month_numeric
    return None, None, None

print(data.columns)

data[['Times', 'FormattedDate', 'MonthNumeric']] = data['Time'].apply(preprocess_time).apply(pd.Series)
print(data['Times'])

import numpy as np
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load and preprocess your data as shown in the previous code snippets

# Splitting data into training and testing sets
train_size = int(0.8 * len(data))
train_data = data.iloc[:train_size]
test_data = data.iloc[train_size:]

# Fit ARIMA model
order = (5, 1, 0)  # Example order (p, d, q)
model = ARIMA(train_data['TrafficCount'], order=order)
model_fit = model.fit(disp=0)

# Make predictions
predictions = model_fit.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1, typ='levels')

# Evaluate model
mse = mean_squared_error(test_data['TrafficCount'], predictions)
rmse = np.sqrt(mse)
print("Root Mean Squared Error:", rmse)
