import pandas as pd
import os

#cwd_join = os.getcwd() + "\\"
#database_rel = os.path.relpath('/Database/weblog.csv')
#database_abs = cwd_join + database_rel
data = pd.read_csv('weblog.csv')

data['Time'] = pd.to_datetime(data['Time'], format='%d%m%Y', errors='ignore')

data['URL'] = data['URL'].map(lambda x: x.lstrip('0'))
data.describe()

data['month'] = data['Time'].str.slice(3, 6)
data['day'] = data['Time'].str.slice(0, 2)

data['day'] = pd.to_numeric(data['day'], errors='coerce')  # Convert non-numeric values to NaN
data['day'] = data['day'].clip(lower=1, upper=30)

data['Methods'] = data['URL'].str.split('/').str[0]


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
st.title("* Example\n---")
print(data["Time"].head())

# Create features: IP, Time, Status
data['IP'] = data['IP']
data['TimeInterval'] = data['Time']  # You can adjust the time interval granularity
data['Status'] = data['Status']


from scipy.stats import zscore


# Right-side content (Traffic Anomaly Detection)
col1, col2, col3 = st.columns(3)

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

global grouped

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

with col3:
    from statsmodels.tsa.arima.model import ARIMA

    import pandas as pd
    import numpy as np
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    import streamlit as st

    log_data = pd.read_csv('weblog.csv')
    print(log_data.head())

    st.markdown("<h1 style='text-align: center;'>Forecast</h1>", unsafe_allow_html=True)

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

    total_requests = len(log_data["IP"])  # Total number of requests

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

    # np.array(extracted_data)
    print(extracted_data)

    differenced_data = target_variable.diff().dropna()

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

    days_ahead = list(range(1, pred_no + 1)) # Adjust the range based on the number of days predicted

    from Forecasting_code import denormalized_future_predictions

    # Create a new figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the predicted values
    ax.plot(days_ahead, denormalized_future_predictions, marker='o', linestyle='-', color='b', label='Predicted Values')

    # Set labels and title for the plot
    ax.set_xlabel('Days Ahead')
    ax.set_ylabel('Predicted Requests')
    ax.set_title('Predicted Values for Future Days')

    # Display the legend and grid
    ax.legend()
    ax.grid()

    # Display the plot using Streamlit
    st.pyplot(fig)

    st.success("Successfully loaded the forecast")
