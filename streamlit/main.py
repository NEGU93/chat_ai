import os
from dotenv import load_dotenv
import streamlit as st
import importlib

st.markdown("# CHAT AI")

MODEL_CONFIG = {
    "LLAMA": {
        "model_name": "llama3.2",
        "class_name": "OpenAIWrapper",
        "class_constructor_params": {
            "base_url": 'http://localhost:11434/v1',
            "api_key": 'ollama'
        }
    },
    "OpenAI": {
        "model_name": "gpt-4o-mini",
        "class_name": "OpenAIWrapper",
        "class_constructor_params": {}  # Will use env var
    },
    "Anthropic": {
        "model_name": "claude-3-7-sonnet-latest",
        "class_name": "ClaudeWrapper",
        "class_constructor_params": {}  # Will use env var
    },
    "Google": {
        "model_name": "gemini-2.0-flash",
        "class_name": "OpenAIWrapper",
        "class_constructor_params": {
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/"
        }
    },
}

class OpenAIWrapper:
    def __init__(self, **kwargs):
        import openai
        self.client = openai.OpenAI(**kwargs)

    def chat(self, model_name, prompt):
        return self.client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )

class ClaudeWrapper:
    def __init__(self, **kwargs):
        from anthropic import Anthropic
        self.client = Anthropic(**kwargs)

    def chat(self, model_name, prompt):
        return self.client.messages.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            stream=True
        )

def get_response(model_key, prompt):
    config = MODEL_CONFIG[model_key]
    class_name = config["class_name"]
    model_name = config["model_name"]
    class_constructor_params = config.get("class_constructor_params", {})

    instance = globals()[class_name](**class_constructor_params)
    return instance.chat(model_name, prompt)


def get_available_models():
    load_dotenv(override=True)
    openai_api_key = os.getenv('OPENAI_API_KEY')
    anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
    google_api_key = os.getenv('GOOGLE_API_KEY')
    models = ["LLAMA"]
    if openai_api_key:
        models.append("OpenAI")
    if anthropic_api_key:
        models.append("Anthropic")
    if google_api_key:
        models.append("Google")
        MODEL_CONFIG["Google"]["class_constructor_params"]["api_key"] = google_api_key
    return models

model = st.sidebar.selectbox("Select Model", get_available_models())

prompt = st.text_area("Enter your prompt here", height=200)

response = get_response(prompt=prompt, model_key=model)


if st.button("Generate Response") and prompt:
    with st.spinner("Thinking..."):
        response_area = st.empty()
        full_response = ""

        response = get_response(prompt=prompt, model_key=model)

        # LLAMA + OpenAI streaming case
        if model in ["LLAMA", "OpenAI", "Google"]:
            for chunk in response:
                delta = chunk.choices[0].delta.content if hasattr(chunk.choices[0], "delta") else ""
                if delta:
                    full_response += delta
                    response_area.markdown(full_response)
        
        # Claude streaming case
        elif model == "Anthropic":
            for chunk in response:
                if hasattr(chunk, "content"):
                    full_response += chunk.content
                    response_area.markdown(full_response)