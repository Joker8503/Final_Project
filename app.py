import streamlit as st
from prediction_page import show_predict_page
from explore_page import show_explore_page

# Use Streamlit's sidebar to navigate between pages
page = st.sidebar.selectbox("Make a prediction or Explore Stats:", ["Predict", "Explore"])

if page == "Predict":
    show_predict_page()
else:
    show_explore_page()


