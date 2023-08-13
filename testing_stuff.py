import requests
import pandas as pd

# Replace this with the raw URL of the dataset file on GitHub
raw_url = "https://raw.huzaifacoder/WebLogAnalysis-Using-AI/blob/main/weblog.csv"

# Fetch data from the raw URL
response = requests.get(raw_url, verify=False)
# Check if the request was successful
if response.status_code == 200:
    # Load the data using Pandas (assuming it's a CSV file)
    data = pd.read_csv(response.text)

    # Now you can work with the 'data' DataFrame
    print(data.head())
else:
    print("Failed to fetch data")
