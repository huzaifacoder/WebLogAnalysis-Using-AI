import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from datetime import datetime
import os
#  read csv file
#duplicate = pd.read_csv('\\Database\\weblog.csv', squeeze=True)

cwd_join = os.getcwd() + "\\"
database_rel = os.path.relpath('Database\\weblog.csv')
database_abs = cwd_join + database_rel
data = pd.read_csv(database_abs)


data['Time'] = data['Time'].map(lambda x: x.lstrip('['))
data['Time'] = data['Time'].str.split(':', n = 1, expand = True)

data['Time'] = pd.to_datetime(data['Time'], format='%d%m%Y', errors='ignore')

data = data.rename(columns={'Staus': 'Status'}, index={'ONE': 'one'})
data['URL'] = data['URL'].map(lambda x: x.lstrip('0'))
data.describe()


data['month'] = data['Time'].str.slice(3, 6)
data['day'] = data['Time'].str.slice(0, 2)


data['Methods'] = data['URL'].str.split('/').str[0]

#if data['URL'].str.contains('.js').any():
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
data['Methods'].value_counts().head(40).plot.bar(color = color)
plt.title('Most Popular Methods by the Users', fontsize = 20)
plt.show()


plt.rcParams['figure.figsize'] = (18, 7)

color = plt.cm.copper(np.linspace(0, 1, 40))
data['month'].value_counts().head(40).plot.bar(color = 'cyan')
plt.title('Most Popular Months of Logins', fontsize = 20)
plt.show()

plt.rcParams['figure.figsize'] = (18, 7)

color = plt.cm.copper(np.linspace(0, 1, 40))
data['day'].value_counts().head(40).plot.bar(color = 'tomato')
plt.title('Most Popular Days of Logins', fontsize = 20)
plt.show()

plt.rcParams['figure.figsize'] = (18, 7)

color = plt.cm.Wistia(np.linspace(0, 1, 40))
data['Status'].value_counts().head(40).plot.bar(color = 'seagreen')
plt.title('Most Popular statuses for the Users', fontsize = 20)
plt.show()


import streamlit as st

st.set_page_config(page_title = "Web Log Analysis", page_icon = ":desktop_computer:")


# Function to display Status distribution plot
def display_status_distribution():
    st.markdown("## Status Distribution")
    st.bar_chart(data['Status'].value_counts().head(40))


# Function to display Day-wise distribution plot
def display_daywise_distribution():
    st.markdown("## Day-wise Distribution")
    st.bar_chart(data['day'].value_counts().head(40))


# Function to display Month-wise distribution plot
def display_monthwise_distribution():
    st.markdown("## Month-wise Distribution")
    st.bar_chart(data['month'].value_counts().head(40))


# Function to display HTTP Methods distribution plot
def display_methods_distribution():
    st.markdown("## HTTP Methods Distribution")
    st.bar_chart(data['Methods'].value_counts().head(40))

# Sidebar navigation
selected_page = st.sidebar.radio("Select a page:",
    ("Status Distribution", "Day-wise Distribution", "Month-wise Distribution", "HTTP Methods Distribution"))

# Display selected page content
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

# Sidebar content
st.sidebar.subheader("Anomaly Detection Settings")
threshold = st.sidebar.number_input("Threshold for Z-Score Anomaly Detection", value=2.5)

# Main content
st.title("Traffic Anomaly Detection")

if st.button("Detect Anomalies"):
    # Calculate z-scores for IP traffic counts within each time interval and status
    grouped = data.groupby(['TimeInterval', 'IP', 'Status']).size().reset_index(name='TrafficCount')
    grouped['ZScore'] = grouped.groupby(['TimeInterval', 'Status'])['TrafficCount'].transform(zscore)

    # Identify anomalies based on threshold
    anomalies = grouped[grouped['ZScore'] > threshold]

    # Display anomalies
    st.subheader("Detected Anomalies:")
    st.write(anomalies)

    # Visualize anomalies
    # ... You can create visualizations to show anomalies on time series plots or other relevant charts
