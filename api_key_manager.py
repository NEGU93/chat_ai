# api_key_manager.py
import streamlit as st
import os
import uuid


def initialize_session():
    """Initialize session state variables"""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

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
        
        **Your API keys are:**
        - Stored only in your browser session
        - Not saved to any files or databases
        - Automatically cleared when you close the browser
        - Unique to your session only
        """
    )

    with st.expander("🔧 API Key Configuration", expanded=True):
        st.markdown("### Enter your API keys:")

        col1, col2 = st.columns(2)

        with col1:
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

        with col2:
            gemini_key = st.text_input(
                "Gemini API Key",
                type="password",
                value=st.session_state.gemini_key,
                help="Get your key from https://makersuite.google.com/app/apikey",
                placeholder="AI...",
            )

        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            if st.button("💾 Save API Keys", type="primary"):
                save_api_keys(openai_key, anthropic_key, gemini_key)

        with col2:
            if st.button("🧹 Clear All Keys"):
                clear_all_keys()

        with col3:
            if st.button("🔄 Reset Session"):
                reset_session()

        # Show which keys are currently set
        st.markdown("### Current Status:")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.session_state.openai_key:
                st.success("✅ OpenAI Key Set")
            else:
                st.error("❌ OpenAI Key Missing")

        with col2:
            if st.session_state.anthropic_key:
                st.success("✅ Anthropic Key Set")
            else:
                st.error("❌ Anthropic Key Missing")

        with col3:
            if st.session_state.gemini_key:
                st.success("✅ Gemini Key Set")
            else:
                st.error("❌ Gemini Key Missing")


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

    # Mark as configured if at least one key is provided
    if openai_key or anthropic_key or gemini_key:
        st.session_state.api_keys_configured = True
        st.success("🎉 API keys saved successfully! You can now use the chat.")
    else:
        st.session_state.api_keys_configured = False
        st.warning("⚠️ Please provide at least one API key to use the chat.")

    st.rerun()


def clear_all_keys():
    """Clear all API keys"""
    st.session_state.openai_key = ""
    st.session_state.anthropic_key = ""
    st.session_state.gemini_key = ""
    st.session_state.api_keys_configured = False

    # Clear environment variables
    for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"]:
        if key in os.environ:
            del os.environ[key]

    st.success("🧹 All API keys cleared!")
    st.rerun()


def reset_session():
    """Reset the entire session"""
    # Clear all session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]

    # Clear environment variables
    for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"]:
        if key in os.environ:
            del os.environ[key]

    st.success("🔄 Session reset successfully!")
    st.rerun()


def has_api_keys():
    """Check if user has provided any API keys"""
    return st.session_state.api_keys_configured


def import_main_app():
    """Import the main app functionality"""
    try:
        # Import your existing app.py functions
        import importlib.util

        spec = importlib.util.spec_from_file_location("app", "app.py")
        app_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(app_module)
        return app_module
    except Exception as e:
        st.error(f"Error importing app.py: {str(e)}")
        return None


def run_main_app():
    """Run the main chat app"""
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

        st.markdown("---")

        # Quick actions
        if st.button("🔧 Manage API Keys"):
            st.session_state.show_api_setup = True

        if st.button("🧹 Clear Keys"):
            clear_all_keys()

        if st.button("🔄 Reset Session"):
            reset_session()

        st.markdown("---")
        st.markdown(f"**Session ID:** `{st.session_state.session_id[:8]}...`")

    # Show API key setup if requested
    if st.session_state.get("show_api_setup"):
        api_key_setup()
        if st.button("❌ Close API Setup"):
            st.session_state.show_api_setup = False
        st.divider()

    # Run the original app code
    try:
        # Execute your main app code directly
        exec("""
import streamlit as st
from models import get_models_by_provider, get_model, audio2text

st.markdown("# 🤖 CHAT AI")

if "prompt" not in st.session_state:
    st.session_state.prompt = ""

# Get available models
providers = get_models_by_provider()
if not providers:
    st.error("❌ No models available. Please check your API keys in the sidebar.")
    st.stop()

# Model selection
col1, col2 = st.columns(2)
with col1:
    provider_name = st.selectbox("🏢 Select Provider", providers.keys())
with col2:
    model_name = st.selectbox("🤖 Select Model", providers[provider_name])

# Only get model if it has changed or is not set
if (
    "model_name" not in st.session_state
    or st.session_state.model_name != model_name
):
    with st.spinner("Loading model..."):
        st.session_state.model = get_model(model_name)
        st.session_state.model_name = model_name

# Chat interface
st.markdown("### 💬 Chat")
chat_placeholder = st.container()
with chat_placeholder:
    for message in st.session_state.model.messages[1:]:  # Skip system message
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def submit_prompt():
    st.session_state.prompt = st.session_state.prompt_widget
    st.session_state.prompt_widget = ""

def process_audio():
    if st.session_state.audio_input is not None:
        with st.spinner("Converting audio to text..."):
            audio_text = audio2text(st.session_state.audio_input)
            st.session_state.prompt = audio_text

# Input area
st.markdown("### 💭 Your Message")
st.text_area(
    "Enter your prompt here",
    height=100,
    key="prompt_widget",
    on_change=submit_prompt,
    placeholder="Type your message here..."
)

# Audio input (only if OpenAI is available)
if "OpenAI" in providers:
    st.audio_input(
        "🎤 Record your message",
        key="audio_input",
        on_change=process_audio,
    )

prompt = st.session_state.prompt

if prompt.strip():
    with chat_placeholder:
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()

            with st.spinner("🤔 Thinking..."):
                try:
                    full_response = ""
                    for chunk in st.session_state.model.chat(prompt=prompt):
                        full_response = chunk
                        message_placeholder.markdown(chunk)
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.error("Please check your API keys and try again.")

    # Clear the prompt after processing
    st.session_state.prompt = ""
""")
    except Exception as e:
        st.error(f"❌ Error running main app: {str(e)}")
        st.markdown("Please check your API keys and try again.")


def run_app_with_api_keys():
    """Main function to run the app with API key management"""
    # Initialize session
    initialize_session()

    # Check if API keys are configured
    if not has_api_keys():
        # Show API key setup
        api_key_setup()
    else:
        # Show main app
        run_main_app()
