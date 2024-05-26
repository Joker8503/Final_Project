import streamlit as st
from prediction_page import show_predict_page

# Use Streamlit's sidebar to navigate between pages
page = st.sidebar.selectbox("Make a prediction or Explore Stats:", ["Predict"])

if page == "Predict":
    show_predict_page()


