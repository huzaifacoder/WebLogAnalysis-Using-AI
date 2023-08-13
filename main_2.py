import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from datetime import datetime

duplicate = pd.read_csv('weblog.csv', squeeze=True)

data = pd.DataFrame(duplicate)


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

st.bar_chart(data['Status'].value_counts().head(40))

#data["day"] = data[pd.to_numeric(data['day'], errors='coerce').notnull()]
st.bar_chart(data['day'].value_counts().head(40))

st.bar_chart(data['month'].value_counts().head(40))

st.bar_chart(data['Methods'].value_counts().head(40))

