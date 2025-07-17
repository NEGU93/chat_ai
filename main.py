# main.py - Your new main file for Hugging Face Spaces
import streamlit as st
from api_key_manager import run_app_with_api_keys

# Configure Streamlit page
st.set_page_config(page_title="Chat AI", page_icon="🤖", layout="wide")

# Run the app with API key management
if __name__ == "__main__":
    run_app_with_api_keys()
