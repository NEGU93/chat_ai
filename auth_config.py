# auth_config.py
import streamlit as st
import hashlib
import os
from datetime import datetime, timedelta
import time


class AuthManager:
    def __init__(self):
        self.max_login_attempts = 3
        self.lockout_duration = 300  # 5 minutes

        # Initialize users in session state instead of file
        if "users_db" not in st.session_state:
            st.session_state.users_db = self.get_default_users()

    def hash_password(self, password):
        """Hash password with salt"""
        salt = "streamlit_chat_app_salt"  # In production, use random salt per user
        return hashlib.sha256((password + salt).encode()).hexdigest()

    def get_default_users(self):
        """Get default users"""
        return {
            "demo": {
                "password": self.hash_password("demo123"),
                "created_at": datetime.now().isoformat(),
                "failed_attempts": 0,
                "locked_until": None,
            }
        }

    def load_users(self):
        """Load users from session state"""
        return st.session_state.users_db

    def save_users(self, users):
        """Save users to session state"""
        st.session_state.users_db = users

    def is_user_locked(self, username):
        """Check if user is locked out"""
        users = self.load_users()
        if username not in users:
            return False

        locked_until = users[username].get("locked_until")
        if locked_until:
            locked_time = datetime.fromisoformat(locked_until)
            if datetime.now() < locked_time:
                return True
            else:
                # Unlock user
                users[username]["locked_until"] = None
                users[username]["failed_attempts"] = 0
                self.save_users(users)

        return False

    def authenticate_user(self, username, password):
        """Authenticate user with lockout protection"""
        if self.is_user_locked(username):
            return (
                False,
                "Account temporarily locked due to too many failed attempts",
            )

        users = self.load_users()

        if username not in users:
            return False, "Invalid username or password"

        user = users[username]

        if self.hash_password(password) == user["password"]:
            # Reset failed attempts on successful login
            user["failed_attempts"] = 0
            user["locked_until"] = None
            user["last_login"] = datetime.now().isoformat()
            self.save_users(users)
            return True, "Login successful"
        else:
            # Increment failed attempts
            user["failed_attempts"] = user.get("failed_attempts", 0) + 1

            if user["failed_attempts"] >= self.max_login_attempts:
                # Lock account
                user["locked_until"] = (
                    datetime.now() + timedelta(seconds=self.lockout_duration)
                ).isoformat()
                self.save_users(users)
                return (
                    False,
                    f"Account locked for {self.lockout_duration // 60} minutes due to too many failed attempts",
                )

            self.save_users(users)
            remaining_attempts = (
                self.max_login_attempts - user["failed_attempts"]
            )
            return (
                False,
                f"Invalid username or password. {remaining_attempts} attempts remaining",
            )

    def register_user(self, username, password, confirm_password):
        """Register new user"""
        if password != confirm_password:
            return False, "Passwords do not match"

        if len(password) < 8:
            return False, "Password must be at least 8 characters long"

        users = self.load_users()

        if username in users:
            return False, "Username already exists"

        users[username] = {
            "password": self.hash_password(password),
            "created_at": datetime.now().isoformat(),
            "failed_attempts": 0,
            "locked_until": None,
        }

        self.save_users(users)
        return True, "User registered successfully"


# Import your existing app functions
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


def api_key_setup():
    """API Key configuration UI"""
    st.markdown("## 🔑 API Key Setup")
    st.markdown(
        "Please provide your API keys to use the AI models. Your keys are stored securely in your session and are not saved."
    )

    with st.expander("API Key Configuration", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            openai_key = st.text_input(
                "OpenAI API Key",
                type="password",
                value=st.session_state.get("openai_key", ""),
                help="Get your key from https://platform.openai.com/api-keys",
            )

            anthropic_key = st.text_input(
                "Anthropic API Key",
                type="password",
                value=st.session_state.get("anthropic_key", ""),
                help="Get your key from https://console.anthropic.com/",
            )

        with col2:
            gemini_key = st.text_input(
                "Gemini API Key",
                type="password",
                value=st.session_state.get("gemini_key", ""),
                help="Get your key from https://makersuite.google.com/app/apikey",
            )

        if st.button("Save API Keys"):
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

            st.success("API keys saved successfully!")
            st.rerun()


def has_api_keys():
    """Check if user has provided any API keys"""
    return (
        st.session_state.get("openai_key")
        or st.session_state.get("anthropic_key")
        or st.session_state.get("gemini_key")
    )


def logout():
    """Clear session and logout user"""
    # Clear session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]

    # Clear environment variables
    for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"]:
        if key in os.environ:
            del os.environ[key]

    st.rerun()


def run_main_app():
    """Run the main chat app with API key management"""
    # Check if user has provided API keys
    if not has_api_keys():
        api_key_setup()
        return

    # Show current API key status in sidebar
    with st.sidebar:
        st.markdown("### API Key Status")
        if st.session_state.get("openai_key"):
            st.success("✅ OpenAI")
        if st.session_state.get("anthropic_key"):
            st.success("✅ Anthropic")
        if st.session_state.get("gemini_key"):
            st.success("✅ Gemini")

        if st.button("Update API Keys"):
            st.session_state.show_api_setup = True

        st.markdown("---")
        st.markdown(f"**Logged in as:** {st.session_state.username}")
        if st.button("Logout"):
            logout()

    # Show API key setup if requested
    if st.session_state.get("show_api_setup"):
        api_key_setup()
        if st.button("Hide API Key Setup"):
            st.session_state.show_api_setup = False
        st.divider()

    # Now run your original app code
    try:
        # Import and execute your main app
        app_module = import_main_app()
        if app_module:
            # Execute the main app code (everything except imports)
            exec("""
import streamlit as st
from models import get_models_by_provider, get_model, audio2text

st.markdown("# CHAT AI")

if "prompt" not in st.session_state:
    st.session_state.prompt = ""

# Sidebar for model selection
providers = get_models_by_provider()
if not providers:
    st.error("No models available. Please check your API keys.")
    st.stop()

provider_name = st.sidebar.selectbox("Select Provider", providers.keys())
model_name = st.sidebar.selectbox("Select Model", providers[provider_name])

# Only get model if it has changed or is not set
if (
    "model_name" not in st.session_state
    or st.session_state.model_name != model_name
):
    st.session_state.model = get_model(model_name)
    st.session_state.model_name = model_name

# Chat
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
        # Convert audio to text
        audio_text = audio2text(st.session_state.audio_input)
        st.session_state.prompt = audio_text

st.text_area(
    "Enter your prompt here",
    height=200,
    key="prompt_widget",
    on_change=submit_prompt,
)

if "OpenAI" in providers:
    st.audio_input(
        "Record your message",
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

            with st.spinner("Thinking..."):
                for chunk in st.session_state.model.chat(prompt=prompt):
                    message_placeholder.markdown(chunk)

    # Clear the prompt after processing
    st.session_state.prompt = ""
""")
    except Exception as e:
        st.error(f"Error running main app: {str(e)}")
        st.markdown("Please check your API keys and try again.")


# Enhanced main app with registration
def enhanced_auth_app():
    auth_manager = AuthManager()

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if "show_register" not in st.session_state:
        st.session_state.show_register = False

    if not st.session_state.authenticated:
        st.markdown("# 🔐 Access Chat AI")

        tab1, tab2 = st.tabs(["Login", "Register"])

        with tab1:
            st.markdown("### Login to your account")
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submit = st.form_submit_button("Login")

                if submit:
                    if username and password:
                        success, message = auth_manager.authenticate_user(
                            username, password
                        )
                        if success:
                            st.session_state.authenticated = True
                            st.session_state.username = username
                            st.success(message)
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(message)
                    else:
                        st.error("Please enter both username and password")

        with tab2:
            st.markdown("### Create a new account")
            with st.form("register_form"):
                new_username = st.text_input("Choose Username")
                new_password = st.text_input(
                    "Choose Password", type="password"
                )
                confirm_password = st.text_input(
                    "Confirm Password", type="password"
                )
                register = st.form_submit_button("Register")

                if register:
                    if new_username and new_password and confirm_password:
                        success, message = auth_manager.register_user(
                            new_username, new_password, confirm_password
                        )
                        if success:
                            st.success(message)
                            st.info("You can now login with your new account")
                        else:
                            st.error(message)
                    else:
                        st.error("Please fill in all fields")

        st.markdown("---")
        st.markdown("**Demo Account:** Username: `demo`, Password: `demo123`")

    else:
        # Show main app
        run_main_app()


# Usage instructions
"""
To use this enhanced authentication:

1. Replace your main code with the enhanced version above
2. Add the AuthManager class to your app
3. The app will create a users.json file to store user data
4. Users can register their own accounts
5. Failed login attempts are tracked and accounts get locked temporarily

For production deployment:
- Use environment variables for sensitive data
- Consider using a proper database instead of JSON files
- Add email verification for registration
- Use secure session management
- Add password reset functionality
- Consider OAuth integration (Google, GitHub, etc.)
"""
