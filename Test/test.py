import pandas as pd

# Sample data
data = {
    'Date': [
        '01/Dec/2017', '01/Mar/2018', '02/Dec/2017', '02/Mar/2018',
        '03/Dec/2017', '07/Nov/2017', '08/Nov/2017', '09/Nov/2017',
        '10/Nov/2017', '11/Nov/2017', '12/Dec/2017', '12/Nov/2017',
        '13/Dec/2017', '13/Nov/2017', '14/Dec/2017'
    ],
}

# Create a DataFrame from the data
df = pd.DataFrame(data)

# Split the 'Date' column into 'Date' and 'Time' components


# Convert the 'Date' column to a datetime format
df['Date'] = pd.to_datetime(df['Date'], format='%d/%b/%Y')

# Sort the DataFrame by the 'Date' column
df = df.sort_values(by='Date')

# Convert the sorted 'Date' column to the desired format (e.g., 2018121)
df['Date'] = df['Date'].dt.strftime('%Y%m%d')

df['Index'] = df["Date"].index

# Display the sorted DataFrame
print(df)
