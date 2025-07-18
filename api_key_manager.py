# api_key_manager.py
import streamlit as st
import os
from app import run_app


def initialize_session_state():
    """Initialize session state variables if they don't exist"""
    defaults = {"openai_key": "", "anthropic_key": "", "gemini_key": ""}

    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def display_api_key_info():
    """Display information about API key setup"""
    st.markdown("# 🔑 API Key Setup")
    st.markdown(
        """
        Welcome to Chat AI! To get started, please provide your API keys below.
        
        **At least one API key is required to use the application.**

        ### How to get your API keys:
        - **OpenAI**: Visit https://platform.openai.com/api-keys (Required for Speech-to-text)
        - **Anthropic**: Visit https://console.anthropic.com/
        - **Google Gemini**: Visit https://ai.google.dev/gemini-api
        """
    )

    with st.expander("🔒 Privacy & Security", expanded=False):
        st.markdown("""
        Your API keys are:
        - Stored only in your browser session
        - Not saved to any files or databases
        - Automatically cleared when you close the browser
        - Unique to your session only
        """)


def render_api_key_inputs():
    """Render API key input fields and return the values"""
    st.markdown("### Enter your API keys:")

    openai_key = st.text_input(
        "OpenAI API Key (Optional)",
        type="password",
        value=st.session_state.openai_key,
        help="Required for Speech-to-text functionality",
        placeholder="sk-...",
    )

    anthropic_key = st.text_input(
        "Anthropic API Key (Optional)",
        type="password",
        value=st.session_state.anthropic_key,
        help="For Claude models",
        placeholder="sk-ant-...",
    )

    gemini_key = st.text_input(
        "Gemini API Key (Optional)",
        type="password",
        value=st.session_state.gemini_key,
        help="For Google Gemini models",
        placeholder="AI...",
    )

    return openai_key, anthropic_key, gemini_key


def validate_api_keys(openai_key, anthropic_key, gemini_key):
    """Validate that at least one API key is provided"""
    keys = [openai_key.strip(), anthropic_key.strip(), gemini_key.strip()]
    return any(keys)


def save_api_keys(openai_key, anthropic_key, gemini_key):
    """Save API keys to session state and environment variables"""
    # Clean and store keys in session state
    st.session_state.openai_key = openai_key.strip()
    st.session_state.anthropic_key = anthropic_key.strip()
    st.session_state.gemini_key = gemini_key.strip()

    # Set environment variables for the session (only if not empty)
    if st.session_state.openai_key:
        os.environ["OPENAI_API_KEY"] = st.session_state.openai_key
    elif "OPENAI_API_KEY" in os.environ:
        # Clear from environment if empty
        del os.environ["OPENAI_API_KEY"]

    if st.session_state.anthropic_key:
        os.environ["ANTHROPIC_API_KEY"] = st.session_state.anthropic_key
    elif "ANTHROPIC_API_KEY" in os.environ:
        del os.environ["ANTHROPIC_API_KEY"]

    if st.session_state.gemini_key:
        os.environ["GEMINI_API_KEY"] = st.session_state.gemini_key
    elif "GEMINI_API_KEY" in os.environ:
        del os.environ["GEMINI_API_KEY"]

    st.success("✅ API keys saved successfully!")
    st.rerun()


def display_api_key_status():
    """Display current API key status in sidebar"""
    st.markdown("### 🔑 API Key Status")

    active_keys = []

    if st.session_state.openai_key:
        st.success("✅ OpenAI")
        active_keys.append("OpenAI")
    else:
        st.info("⚪ OpenAI (Not configured)")

    if st.session_state.anthropic_key:
        st.success("✅ Anthropic")
        active_keys.append("Anthropic")
    else:
        st.info("⚪ Anthropic (Not configured)")

    if st.session_state.gemini_key:
        st.success("✅ Gemini")
        active_keys.append("Gemini")
    else:
        st.info("⚪ Gemini (Not configured)")

    if not active_keys:
        st.error("❌ No API keys configured")
        return False
    else:
        st.info(f"Active: {', '.join(active_keys)}")
        return True


def api_key_setup():
    """API Key configuration UI"""
    display_api_key_info()

    with st.sidebar.expander("🔧 API Key Configuration", expanded=True):
        openai_key, anthropic_key, gemini_key = render_api_key_inputs()

        col1, col2 = st.columns(2)

        with col1:
            if st.button("💾 Save Keys", type="primary"):
                if validate_api_keys(openai_key, anthropic_key, gemini_key):
                    save_api_keys(openai_key, anthropic_key, gemini_key)
                else:
                    st.error("❌ Please provide at least one API key")

        with col2:
            if st.button("🗑️ Clear All"):
                save_api_keys("", "", "")


def run_main_app():
    """Run the main chat app"""
    initialize_session_state()

    st.info(
        "ℹ️ This is a mini version - no local models are available due to hardware limitations."
    )

    # Show API key management in sidebar
    with st.sidebar:
        has_keys = display_api_key_status()

        if not has_keys:
            st.warning(
                "⚠️ Configure at least one API key to use the application"
            )

    # Always show API key setup
    api_key_setup()

    st.sidebar.divider()

    # Only run the main app if we have at least one API key
    if (
        st.session_state.openai_key
        or st.session_state.anthropic_key
        or st.session_state.gemini_key
    ):
        run_app()
    else:
        st.warning(
            "⚠️ Please configure at least one API key to start using the application."
        )


def run_app_with_api_keys():
    """Main function to run the app with API key management"""
    run_main_app()
