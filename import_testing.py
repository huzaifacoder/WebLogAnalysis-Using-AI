import streamlit as st
import pandas as pd
import re

# Set the title of the Streamlit app
st.title("Log File Upload and Parsing")

# Create a file uploader widget
uploaded_file = st.file_uploader("Upload a log file", type=["csv", "xlsx", "xls", "txt", "json", "log"])
global df

if uploaded_file is not None:
    # Check the file format and read the data accordingly
    if uploaded_file.type == "application/vnd.ms-excel":
        df = pd.read_excel(uploaded_file)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        df = pd.read_excel(uploaded_file)
    elif uploaded_file.type == "text/csv":
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.type == "text/plain":
        # Check if it's an Apache log file (you may need to adjust the regex pattern)
        if re.search(r'\[.*\] ".+" \d+ \d+', uploaded_file.getvalue().decode("utf-8")):
            # Parse the Apache log data (adjust the parsing logic)
            log_lines = uploaded_file.getvalue().decode("utf-8").split('\n')
            log_data = []
            for line in log_lines:
                parts = re.split(r'\s+', line, maxsplit=7)
                if len(parts) >= 8:
                    log_data.append(parts)
            df = pd.DataFrame(log_data, columns=["IP", "Timestamp", "Request", "Status"])
        else:
            # Handle it as a plain text file
            df = pd.read_csv(uploaded_file, delimiter='\t')
    elif uploaded_file.type == "application/json":
        df = pd.read_json(uploaded_file)
    else:
        st.error("Unsupported file format")

    # Display the data
    st.dataframe(df)