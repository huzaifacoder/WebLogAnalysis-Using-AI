import streamlit as st
import pandas as pd  # For data storage (you can choose a different method as well)


st.title("Feedback Form")

# Get user's feedback
feedback = st.text_area("Please enter your feedback:")

# Get user's rating (example using selectbox)
rating = st.selectbox("Rate our app:", ["1", "2", "3", "4", "5"])

if st.button("Submit"):
    # Save the feedback to a data structure or storage of your choice
    feedback_data = pd.DataFrame({"Feedback": [feedback], "Rating": [rating]})
    # Set a session state flag to indicate that feedback data exists
    st.session_state.feedback_data_exists = True

    # Append feedback to an existing CSV file or save it to a database
    feedback_data.to_csv("feedback.csv", mode="a", index=False, header=not st.session_state.feedback_data_exists)

    st.success("Thank you for your feedback!")
