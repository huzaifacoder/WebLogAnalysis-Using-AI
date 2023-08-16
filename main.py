import numpy as np
import pandas as pd
import os

cwd_join = os.getcwd() + "\\"
database_rel = os.path.relpath('Database\\weblog.csv')
database_abs = cwd_join + database_rel
data = pd.read_csv(database_abs)

data['Time'] = data['Time'].map(lambda x: x.lstrip('['))
data['Time'] = data['Time'].str.split(':', n=1, expand=True)

data['Time'] = pd.to_datetime(data['Time'], format='%d%m%Y', errors='ignore')

data = data.rename(columns={'Staus': 'Status'}, index={'ONE': 'one'})
data['URL'] = data['URL'].map(lambda x: x.lstrip('0'))
data.describe()

data['month'] = data['Time'].str.slice(3, 6)
data['day'] = data['Time'].str.slice(0, 2)

data['Methods'] = data['URL'].str.split('/').str[0]

# if data['URL'].str.contains('.js').any():
#   data['URL_new'] = data['URL'].str.split('/').str[3]
if data['URL'].str.contains('.php').any():
    data['URL_new'] = data['URL'].str.split('/').str[1]
elif data['URL'].str.contains('.js').any():
    data['URL_new'] = data['URL'].str.split('/').str[3]

print(data)

# Detecting Malicious Activity
malicious_extensions = ['php?id=', '.exe']  # Example malicious extensions


# Define a function to check for malicious activity
def is_malicious(x):
    if isinstance(x, str):  # Check if x is a string
        return any(ext in x for ext in malicious_extensions)
    return False


data['Malicious'] = data['URL_new'].apply(is_malicious)

# Count the occurrences of malicious URLs
malicious_counts = data[data['Malicious']].groupby('URL_new').size().reset_index(name='Count')

# Display URLs with high occurrence counts
print("Malicious URLs with high occurrence counts:")
print(malicious_counts.sort_values(by='Count', ascending=False).head(10))

print(data['day'].unique())

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (18, 7)

color = plt.cm.copper(np.linspace(0, 1, 40))
data['Methods'].value_counts().head(40).plot.bar(color=color)
plt.title('Most Popular Methods by the Users', fontsize=20)
plt.show()

plt.rcParams['figure.figsize'] = (18, 7)

color = plt.cm.copper(np.linspace(0, 1, 40))
data['month'].value_counts().head(40).plot.bar(color='cyan')
plt.title('Most Popular Months of Logins', fontsize=20)
plt.show()

plt.rcParams['figure.figsize'] = (18, 7)

color = plt.cm.copper(np.linspace(0, 1, 40))
data['day'].value_counts().head(40).plot.bar(color='tomato')
plt.title('Most Popular Days of Logins', fontsize=20)
plt.show()

plt.rcParams['figure.figsize'] = (18, 7)

color = plt.cm.Wistia(np.linspace(0, 1, 40))
data['Status'].value_counts().head(40).plot.bar(color='seagreen')
plt.title('Most Popular statuses for the Users', fontsize=20)
plt.show()

import streamlit as st

st.set_page_config(page_title="Web Log Analysis", page_icon=":desktop_computer:")


# Function to display Status distribution plot
def display_status_distribution():  # I WANT TO MAKE IT Options no radio
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

select_pages_list = ["Status Distribution", "Day-wise Distribution", "Month-wise Distribution","HTTP Methods Distribution"]

# Display selected page content
st.title("Web log Analysis \n---")
st.markdown("## Basic Statistics")


# Sidebar navigation
selected_page = st.selectbox("", select_pages_list)

if selected_page == "Status Distribution":
    display_status_distribution()
elif selected_page == "Day-wise Distribution":
    display_daywise_distribution()
elif selected_page == "Month-wise Distribution":
    display_monthwise_distribution()
elif selected_page == "HTTP Methods Distribution":
    display_methods_distribution()

import pandas as pd
import numpy as np
import streamlit as st
from scipy.stats import zscore

print(data["Time"].head())

# Create features: IP, Time, Status
data['IP'] = data['IP']
data['TimeInterval'] = data['Time']  # You can adjust the time interval granularity
data['Status'] = data['Status']

from scipy.stats import zscore

import streamlit_vertical_slider as svs

threshold = st.slider("Threshold for Z-Score Anomaly Detection", min_value=1.0, max_value=1.9, value=1.5, step=0.05)
st.write("Best below 1.6 threshold")
# Main content
st.markdown("## Traffic Anomaly Detection")


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
    st.subheader(f"IP with the Most TrafficCount: {ip_with_most_traffic} withn the given treshold")
