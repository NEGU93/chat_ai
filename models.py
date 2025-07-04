import os
import subprocess
from typing import Any, Dict, List, Generator
from dotenv import load_dotenv
import openai
from anthropic import Anthropic


def ensure_llama_model(model_name="llama3.2", auto_pull=False) -> bool:
    try:
        # Check available models
        result = subprocess.run(
            ["ollama", "list"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:
            print("[ollama list] Error:", result.stderr.strip())
            return False

        available_models = [
            line.split()[0].split(":")[0].lower()
            for line in result.stdout.strip().splitlines()[1:]  # skip header
            if line.strip()
        ]

        if model_name.lower() in available_models:
            return True

        if auto_pull:
            print(f"Model '{model_name}' not found. Attempting to pull...")
            pull_result = subprocess.run(["ollama", "pull", model_name], check=True)
            return pull_result.returncode == 0

        return False

    except FileNotFoundError:
        print("Ollama is not installed or not in PATH.")
    except subprocess.CalledProcessError as e:
        print("Subprocess error:", e)
    except Exception as e:
        print("Unknown error:", e)

    return False


class OpenAIWrapper:
    def __init__(self, **kwargs: Any):
        """Initialize OpenAI-compatible client."""
        
        self.client = openai.OpenAI(**kwargs)

    def chat(self, model_name: str, prompt: str) -> Generator:
        """
        Generate a streaming response from an OpenAI-compatible model.

        Args:
            model_name (str): The model name to use.
            prompt (str): The prompt to send.

        Returns:
            Generator: A stream of completion chunks.
        """
        return self.client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )


class ClaudeWrapper:
    def __init__(self, **kwargs: Any):
        """Initialize Claude (Anthropic) client."""
        
        self.client = Anthropic(**kwargs)

    def chat(self, model_name: str, prompt: str) -> Generator:
        """
        Generate a streaming response from Claude.

        Args:
            model_name (str): The Claude model to use.
            prompt (str): The input prompt.

        Returns:
            Generator: A stream of Claude response content.
        """
        return self.client.messages.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            stream=True
        )
    
# Model configuration for different providers
MODEL_CONFIG: Dict[str, Dict[str, Any]] = {
    "LLAMA": {
        "model_name": "llama3.2",
        "class_name": "OpenAIWrapper",
        "class_constructor_params": {
            "base_url": 'http://localhost:11434/v1',
            "api_key": 'ollama'
        }
    },
    "ChatGPT": {
        "model_name": "gpt-4o-mini",
        "class_name": "OpenAIWrapper",
        "class_constructor_params": {}
    },
    "Claude": {
        "model_name": "claude-3-7-sonnet-latest",
        "class_name": "ClaudeWrapper",
        "class_constructor_params": {}
    },
    "Gemini": {
        "model_name": "gemini-2.0-flash",
        "class_name": "OpenAIWrapper",
        "class_constructor_params": {
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/"
        }
    },
}


def get_response(model_key: str, prompt: str) -> Generator:
    """
    Get streaming response from the selected model.

    Args:
        model_key (str): The model key (e.g., "ChatGPT", "Claude").
        prompt (str): The input prompt.

    Returns:
        Generator: A generator yielding response chunks.
    """
    config = MODEL_CONFIG[model_key]
    class_name = config["class_name"]
    model_name = config["model_name"]
    class_constructor_params = config.get("class_constructor_params", {})

    instance = globals()[class_name](**class_constructor_params)
    return instance.chat(model_name, prompt)


def get_available_models() -> List[str]:
    """
    Detect available models based on environment variables.

    Returns:
        List[str]: List of available model keys.
    """
    load_dotenv(override=True)
    openai_api_key = os.getenv('OPENAI_API_KEY')
    anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
    google_api_key = os.getenv('GOOGLE_API_KEY')

    models = []
    if ensure_llama_model("llama3.2", auto_pull=False):
        models.append("LLAMA")
    if openai_api_key:
        models.append("ChatGPT")
    if anthropic_api_key:
        models.append("Claude")
    if google_api_key:
        models.append("Gemini")
        MODEL_CONFIG["Gemini"]["class_constructor_params"]["api_key"] = google_api_key
    return models