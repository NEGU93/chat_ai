import streamlit as st
from models import get_available_models, get_response

st.markdown("# CHAT AI")


# Streamlit UI
model = st.sidebar.selectbox("Select Model", get_available_models())

st.markdown(f"Using Model: **{model}**")

prompt = st.text_area("Enter your prompt here", height=200)

if st.button("Generate Response") and prompt:
    with st.spinner("Thinking..."):
        response_area = st.empty()
        full_response = ""

        response = get_response(prompt=prompt, model_key=model)

        # OpenAI-compatible models (ChatGPT, Gemini, LLAMA)
        if model in ["LLAMA", "ChatGPT", "Gemini"]:
            for chunk in response:
                delta = getattr(chunk.choices[0].delta, "content", "")
                if delta:
                    full_response += delta
                    response_area.markdown(full_response)

        # Claude-specific streaming
        elif model == "Claude":
            for chunk in response:
                # Each chunk is a MessageStreamEvent, e.g., ContentBlockDelta
                if hasattr(chunk, "delta") and hasattr(chunk.delta, "text"):
                    full_response += chunk.delta.text
                    response_area.markdown(full_response)
