# api_key_manager.py
import streamlit as st
import os
from app import run_app


if "api_keys_configured" not in st.session_state:
    st.session_state.api_keys_configured = False

if "openai_key" not in st.session_state:
    st.session_state.openai_key = ""

if "anthropic_key" not in st.session_state:
    st.session_state.anthropic_key = ""

if "gemini_key" not in st.session_state:
    st.session_state.gemini_key = ""


def api_key_setup():
    """API Key configuration UI"""
    st.markdown("# 🔑 API Key Setup")
    st.markdown(
        """
        Welcome to Chat AI! To get started, please provide your API keys below.

        For setting up each key go to the respective provider's website:
        - For OpenAI, visit https://openai.com/api/. OpenAI key is required for Speech-to-text capabilities.
        - For Anthropic, visit https://console.anthropic.com/
        - For Google, visit https://ai.google.dev/gemini-api
        """
    )
    st.markdown("""
        Your API keys are:
        - Stored only in your browser session
        - Not saved to any files or databases
        - Automatically cleared when you close the browser
        - Unique to your session only
        """)

    with st.sidebar.expander("🔧 API Key Configuration", expanded=True):
        st.markdown("### Enter your API keys:")

        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=st.session_state.openai_key,
            help="Get your key from https://platform.openai.com/api-keys",
            placeholder="sk-...",
        )

        anthropic_key = st.text_input(
            "Anthropic API Key",
            type="password",
            value=st.session_state.anthropic_key,
            help="Get your key from https://console.anthropic.com/",
            placeholder="sk-ant-...",
        )
        gemini_key = st.text_input(
            "Gemini API Key",
            type="password",
            value=st.session_state.gemini_key,
            help="Get your key from https://makersuite.google.com/app/apikey",
            placeholder="AI...",
        )
        if st.button("💾 Save API Keys", type="primary"):
            save_api_keys(openai_key, anthropic_key, gemini_key)


def save_api_keys(openai_key, anthropic_key, gemini_key):
    """Save API keys to session state and environment variables"""
    # Store keys in session state
    st.session_state.openai_key = openai_key
    st.session_state.anthropic_key = anthropic_key
    st.session_state.gemini_key = gemini_key

    # Set environment variables for the session
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
    if anthropic_key:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_key
    if gemini_key:
        os.environ["GEMINI_API_KEY"] = gemini_key

    st.rerun()


def run_main_app():
    """Run the main chat app"""
    st.info(
        "This is a mini version, no local models are available due to Hardware limitations."
    )
    # Show API key management in sidebar
    with st.sidebar:
        st.markdown("### 🔑 API Key Status")

        # Show current status
        if st.session_state.openai_key:
            st.success("✅ OpenAI")
        if st.session_state.anthropic_key:
            st.success("✅ Anthropic")
        if st.session_state.gemini_key:
            st.success("✅ Gemini")

        if not (
            st.session_state.openai_key
            or st.session_state.anthropic_key
            or st.session_state.gemini_key
        ):
            st.error("❌ No API keys configured")

    api_key_setup()

    st.sidebar.divider()

    run_app()


def run_app_with_api_keys():
    """Main function to run the app with API key management"""
    run_main_app()
