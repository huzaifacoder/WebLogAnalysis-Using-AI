import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("")
st.markdown("<h1 style='text-align: center; font-size: 100px;'>Potential Log 👋</h1>", unsafe_allow_html=True)
st.sidebar.success("Select a Page above")

data = st.sidebar.file_uploader("Upload your file here...")
st.subheader(
    """
    Discover the future with our web log analysis app, specializing in Time Series Analytics. Predict incoming requests. Stay ahead of the curve with data-driven insights.
"""
)
#Your web's potential, unleashed
log_data = pd.read_csv(data)
st.write(log_data)