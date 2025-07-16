import streamlit as st
from models import get_available_models, get_model, audio2text

st.markdown("# CHAT AI")

if "prompt" not in st.session_state:
    st.session_state.prompt = ""

# Streamlit UI
model_name = st.sidebar.selectbox("Select Model", get_available_models())

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

if "OpenAI" in get_available_models():
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
