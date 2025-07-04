import streamlit as st
from models import get_available_models, get_response


if "previous_prompts" not in st.session_state:
    st.session_state.previous_prompts = []
if "assistant_responses" not in st.session_state:
    st.session_state.assistant_responses = []

st.markdown("# CHAT AI")


# Streamlit UI
model = st.sidebar.selectbox("Select Model", get_available_models())

# Chat
chat_placeholder = st.container()
with chat_placeholder:
    for user_msg, assistant_msg in zip(st.session_state.previous_prompts, st.session_state.assistant_responses):
        st.markdown(f"**You:** {user_msg}")
        st.markdown(f"**AI:** {assistant_msg}")
        st.markdown("---")
        # Streamlit placeholder to update the latest message while streaming
        message_placeholder = st.empty()

# Model prompts
prompt = st.text_area("Enter your prompt here", height=200)
if st.button("Send") and prompt.strip():
    with st.spinner("Thinking..."):
        # Start streaming response
        full_response = ""
        response = get_response(
            prompt=prompt,
            model_key=model,
            previous_prompts=st.session_state.previous_prompts,
            assistant_responses=st.session_state.assistant_responses,
        )

        # Stream and accumulate response
        if model in ["LLAMA", "ChatGPT", "Gemini"]:
            for chunk in response:
                delta = getattr(chunk.choices[0].delta, "content", "")
                if delta:
                    full_response += delta
                    message_placeholder.markdown(full_response)
        elif model == "Claude":
            for chunk in response:
                if hasattr(chunk, "delta") and hasattr(chunk.delta, "text"):
                    full_response += chunk.delta.text
                    message_placeholder.markdown(full_response)

        # Update conversation history
        st.session_state.previous_prompts.append(prompt)
        st.session_state.assistant_responses.append(full_response)