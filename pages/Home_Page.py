import streamlit as st
from concurrent.futures import ThreadPoolExecutor

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)



st.write("")
st.markdown("<h1 style='text-align: center; font-size: 100px;'>Potential Log 👋</h1>", unsafe_allow_html=True)
st.sidebar.success("Select a demo above.")

st.subheader(
    """
    Discover the future with our web log analysis app, specializing in Time Series Analytics. Predict incoming requests and optimize performance effortlessly. Stay ahead of the curve with data-driven insights.
"""
)
#Your web's potential, unleashed
