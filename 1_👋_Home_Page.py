import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Potential Log",
    page_icon="👋",
    layout="wide"
)

st.write("")
st.markdown("<h1 style='text-align: center; font-size: 100px;'>Potential Log 👋</h1>", unsafe_allow_html=True)
st.sidebar.success("Select a Page above")

st.subheader(
    """
    Discover the future with our web log analysis app, specializing in Time Series Analytics. Predict incoming requests. Stay ahead of the curve with data-driven insights.
    
    """
)

data = st.file_uploader("Upload your file here...")
try:

    if data is not None:  # Check if a file is uploaded
        # Read the CSV data only if a file is uploaded
        log_data = pd.read_csv(data)

        log_data['Time'] = pd.to_datetime(log_data['Time'], format='%d%m%Y', errors='ignore')

        log_data['URL'] = log_data['URL'].map(lambda x: x.lstrip('0'))

        log_data['month'] = log_data['Time'].str.slice(3, 6)
        log_data['day'] = log_data['Time'].str.slice(0, 2)

        log_data['day'] = pd.to_numeric(log_data['day'], errors='coerce')  # Convert non-numeric values to NaN
        log_data['day'] = log_data['day'].clip(lower=1, upper=30)

        log_data['Methods'] = log_data['URL'].str.split('/').str[0]

        # Function to display Status distribution plot
        def display_status_distribution():
            st.markdown("#### Status Distribution")
            st.bar_chart(log_data["Status"].value_counts().head(40)) # status

        # Function to display Day-wise distribution plot
        def display_daywise_distribution():
            st.markdown("#### Day-wise Distribution")
            st.bar_chart(log_data['day'].value_counts().head(40))

        # Function to display Month-wise distribution plot
        def display_monthwise_distribution():
            st.markdown("#### Month-wise Distribution")
            return st.bar_chart(log_data['month'].value_counts().head(40))

        # Function to display HTTP Methods distribution plot
        def display_methods_distribution():
            st.markdown("#### Methods  Distribution")
            return st.bar_chart(log_data['Methods'].value_counts().head(40))


        select_distribution_list = ["Status Distribution", "Day-wise Distribution", "Month-wise Distribution",
                                    "Methods Distribution"]

        ## Create features: IP, Time, Status
        #log_data['IP'] = log_data[0]
        #log_data['Time'] = log_data[1]  # You can adjust the time interval granularity
        #log_data['Status'] = log_data[3]

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<h1 style='text-align: center;'>Basic Statistics</h1>", unsafe_allow_html=True)

            # Sidebar navigation
            selected_page = st.selectbox("", select_distribution_list)

            if selected_page == "Status Distribution":
                display_status_distribution()
            elif selected_page == "Day-wise Distribution":
                display_daywise_distribution()
            elif selected_page == "Month-wise Distribution":
                display_monthwise_distribution()
            elif selected_page == "Methods Distribution":
                display_methods_distribution()

        with col2:
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

            days_ahead = list(range(1, pred_no + 1))  # Adjust the range based on the number of days predicted

            from Forecasting_code import denormalized_future_predictions

            # Create a new figure and axis
            fig, ax = plt.subplots(figsize=(10, 6))

            # Plot the predicted values
            ax.plot(days_ahead, denormalized_future_predictions, marker='o', linestyle='-', color='b',
                    label='Predicted Values')

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


    else:
        st.info("Please upload a CSV file using the sidebar to see the data.")
except Exception:
    st.error("Invalid CSV file")