# main.py - Your new main file that includes authentication
import streamlit as st
from auth_config import enhanced_auth_app

# Configure Streamlit page
st.set_page_config(
    page_title="Chat AI with Authentication", page_icon="🤖", layout="wide"
)

# Run the app with authentication
if __name__ == "__main__":
    enhanced_auth_app()
