import streamlit as st
from models import get_available_models, get_model

st.markdown("# CHAT AI")

if "prompt" not in st.session_state:
    st.session_state.prompt = ""

# Streamlit UI
model_name = st.sidebar.selectbox("Select Model", get_available_models())

if "model_name" not in st.session_state or st.session_state.model_name != model_name:
    st.session_state.model = get_model(model_name)
    st.session_state.model_name = model_name
# Chat
# Inject CSS styles for chat bubbles alignment
chat_placeholder = st.container()
with chat_placeholder:
    for message in st.session_state.model.messages[1:]: # Skip system message
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def submit_prompt():
    st.session_state.prompt = st.session_state.prompt_widget
    st.session_state.prompt_widget  = ""

st.text_area("Enter your prompt here", height=200, key="prompt_widget", on_change=submit_prompt)
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
