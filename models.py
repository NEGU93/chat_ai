import os
import shutil
import subprocess
from typing import Any, Dict, List, Generator
from collections import defaultdict
from dotenv import load_dotenv
import openai
from anthropic import Anthropic


def ensure_ollama_model(model_name="llama3.2", auto_pull=False) -> bool:
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
            pull_result = subprocess.run(
                ["ollama", "pull", model_name], check=True
            )
            return pull_result.returncode == 0

        return False

    except FileNotFoundError:
        print("Ollama is not installed or not in PATH.")
    except subprocess.CalledProcessError as e:
        print("Subprocess error:", e)
    except Exception as e:
        print("Unknown error:", e)

    return False


class AIWrapper:
    def __init__(self, model_name: str, system_role=None):
        """Initialize the AI wrapper."""
        if system_role is None:
            system_role = (
                "You are a helpful assistant that responds in markdown"
            )
        self.messages = [{"role": "system", "content": system_role}]
        self.model_name = model_name

    def chat(self, prompt: str) -> Generator:
        pass


class OpenAIProvider(AIWrapper):
    def __init__(self, model_name: str, system_role=None, **kwargs: Any):
        """Initialize OpenAI-compatible client."""
        super().__init__(model_name=model_name, system_role=system_role)
        self.client = openai.OpenAI(**kwargs)

    def chat(self, prompt: str) -> Generator:
        """
        Generate a streaming response from an OpenAI-compatible model.

        Args:
            prompt (str): The prompt to send.

        Returns:
            Generator: A stream of completion chunks.
        """
        self.messages.append({"role": "user", "content": prompt})
        stream = self.client.chat.completions.create(
            model=self.model_name, messages=self.messages, stream=True
        )
        result = ""
        for chunk in stream:
            result += chunk.choices[0].delta.content or ""
            yield result
        self.messages.append({"role": "assistant", "content": result})


class OllamaProvider(OpenAIProvider):
    def __init__(self, model_name: str, system_role=None, **kwargs: Any):
        """Initialize Ollama client."""
        if not ensure_ollama_model(model_name, auto_pull=True):
            raise ValueError(
                f"Model '{model_name}' is not available in Ollama."
            )
        super().__init__(
            model_name=model_name, system_role=system_role, **kwargs
        )


class AnthropicProvider(AIWrapper):
    def __init__(self, model_name: str, system_role=None, **kwargs: Any):
        """Initialize Anthropic (Claude) client."""
        super().__init__(model_name=model_name, system_role=system_role)
        self.client = Anthropic(**kwargs)

    def chat(self, prompt: str) -> Generator:
        """
        Generate a streaming response from Claude.

        Args:
            prompt (str): The input prompt.

        Returns:
            Generator: A stream of Claude response content.
        """
        self.messages.append({"role": "user", "content": prompt})
        result = self.client.messages.stream(
            model=self.model_name,
            system=self.messages[0]["content"],
            messages=self.messages[1:],
            max_tokens=1024,
        )
        response = ""
        with result as stream:
            for text in stream.text_stream:
                response += text or ""
                yield response
        self.messages.append({"role": "assistant", "content": response})


class HuggingFaceProvider(AIWrapper):
    def __init__(self, model_name: str, system_role=None, **kwargs: Any):
        """Initialize Hugging Face client."""
        super().__init__(model_name=model_name, system_role=system_role)
        import torch
        from transformers import (
            AutoTokenizer,
            AutoModelForCausalLM,
            BitsAndBytesConfig,
        )

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # self.streamer = TextStreamer(self.tokenizer)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", quantization_config=quant_config
        )

    def chat(self, prompt: str) -> Generator:
        """
        Generate a response from a Hugging Face model.

        Args:
            prompt (str): The input prompt.

        Returns:
            Generator: A stream of response content.
        """
        self.messages.append({"role": "user", "content": prompt})
        inputs = self.tokenizer.apply_chat_template(
            self.messages, return_tensors="pt"
        ).to("cuda")
        outputs = self.model.generate(
            inputs,
            max_new_tokens=2000,  # streamer=self.streamer
        )

        new_tokens = outputs[0][inputs.shape[1] :]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        response = response.replace("assistant", "")
        self.messages.append({"role": "assistant", "content": response})
        yield response


# Single source of truth for all models - only modify this!
MODEL_REGISTRY = {
    # OpenAI Models
    "gpt-4o": {
        "provider": "OpenAI",
        "provider_class": "OpenAIProvider",
        "model_name": "gpt-4o",
        "requires_env": "OPENAI_API_KEY",
        "params": {},
    },
    "gpt-4o-mini": {
        "provider": "OpenAI",
        "provider_class": "OpenAIProvider",
        "model_name": "gpt-4o-mini",
        "requires_env": "OPENAI_API_KEY",
        "params": {},
    },
    "chatgpt": {  # Alias
        "provider": "OpenAI",
        "provider_class": "OpenAIProvider",
        "model_name": "gpt-4o-mini",
        "requires_env": "OPENAI_API_KEY",
        "params": {},
    },
    # Anthropic Models
    "claude-3-5-sonnet": {
        "provider": "Anthropic",
        "provider_class": "AnthropicProvider",
        "model_name": "claude-3-5-sonnet-20241022",
        "requires_env": "ANTHROPIC_API_KEY",
        "params": {},
    },
    "claude-3-haiku": {
        "provider": "Anthropic",
        "provider_class": "AnthropicProvider",
        "model_name": "claude-3-haiku-20240307",
        "requires_env": "ANTHROPIC_API_KEY",
        "params": {},
    },
    "claude": {  # Alias
        "provider": "Anthropic",
        "provider_class": "AnthropicProvider",
        "model_name": "claude-3-5-sonnet-20241022",
        "requires_env": "ANTHROPIC_API_KEY",
        "params": {},
    },
    # Google Models
    "gemini-2.0-flash": {
        "provider": "Google",
        "provider_class": "OpenAIProvider",
        "model_name": "gemini-2.0-flash",
        "requires_env": "GOOGLE_API_KEY",
        "params": {
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/"
        },
    },
    "gemini-1.5-pro": {
        "provider": "Google",
        "provider_class": "OpenAIProvider",
        "model_name": "gemini-1.5-pro",
        "requires_env": "GOOGLE_API_KEY",
        "params": {
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/"
        },
    },
    "gemini": {  # Alias
        "provider": "Google",
        "provider_class": "OpenAIProvider",
        "model_name": "gemini-2.0-flash",
        "requires_env": "GOOGLE_API_KEY",
        "params": {
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/"
        },
    },
    # Ollama Models
    "llama3.2": {
        "provider": "Ollama",
        "provider_class": "OllamaProvider",
        "model_name": "llama3.2",
        "requires_env": None,
        "requires_command": "ollama",
        "params": {
            "base_url": "http://localhost:11434/v1",
            "api_key": "ollama",
        },
    },
    "llama3.1": {
        "provider": "Ollama",
        "provider_class": "OllamaProvider",
        "model_name": "llama3.1",
        "requires_env": None,
        "requires_command": "ollama",
        "params": {
            "base_url": "http://localhost:11434/v1",
            "api_key": "ollama",
        },
    },
    # Hugging Face Models
    "llama-3.1-8b": {
        "provider": "Hugging Face",
        "provider_class": "HuggingFaceProvider",
        "model_name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "requires_env": "HF_TOKEN",
        "params": {},
    },
}


def get_model(model_key: str) -> AIWrapper:
    """
    Get AI model instance for the specified model.

    Args:
        model_key (str): The model key (e.g., "chatgpt", "claude", "gemini").

    Returns:
        AIWrapper: An instance of the appropriate provider for the model.
    """
    if model_key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_key}. Available models: {list(MODEL_REGISTRY.keys())}"
        )

    config = MODEL_REGISTRY[model_key]
    provider_class = config["provider_class"]

    # Build parameters
    params = config["params"].copy()
    params["model_name"] = config["model_name"]

    # Add API key if needed
    if config["requires_env"] and config["requires_env"] in os.environ:
        if config["provider"] == "Google":
            params["api_key"] = os.environ[config["requires_env"]]

    return globals()[provider_class](**params)


def _check_model_availability(config: Dict[str, Any]) -> bool:
    """Check if a model is available based on its requirements."""
    # Check environment variable requirement
    if config.get("requires_env") and not os.getenv(config["requires_env"]):
        return False

    # Check command requirement (like ollama)
    if config.get("requires_command"):
        command = config["requires_command"]
        if shutil.which(command) is None:
            return False
        # For ollama, also check if service is running
        if command == "ollama":
            try:
                result = subprocess.run(
                    ["ollama", "list"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode != 0:
                    return False
            except (subprocess.TimeoutExpired, FileNotFoundError):
                return False

    return True


def get_available_models() -> List[str]:
    """
    Detect available models based on environment variables and system capabilities.

    Returns:
        List[str]: List of available model keys.
    """
    load_dotenv(override=True)

    available_models = []
    for model_key, config in MODEL_REGISTRY.items():
        if _check_model_availability(config):
            available_models.append(model_key)

    return available_models


def get_models_by_provider() -> Dict[str, List[str]]:
    """
    Get available models grouped by provider.

    Returns:
        Dict[str, List[str]]: Dictionary mapping provider names to their available models.
    """
    available_models = get_available_models()
    provider_models = defaultdict(list)

    for model_key in available_models:
        provider = MODEL_REGISTRY[model_key]["provider"]
        provider_models[provider].append(model_key)

    return provider_models


def get_all_models() -> List[str]:
    """Get all registered model keys (regardless of availability)."""
    return list(MODEL_REGISTRY.keys())


def get_all_providers() -> List[str]:
    """Get all registered provider names."""
    return list(set(config["provider"] for config in MODEL_REGISTRY.values()))


def audio2text(audio_input: Any) -> str:
    """
    Convert audio input to text using a speech-to-text service.

    Args:
        audio_input (Any): The audio input to convert.

    Returns:
        str: The transcribed text.
    """
    transcription = openai.OpenAI().audio.transcriptions.create(
        model="whisper-1", file=audio_input, response_format="text"
    )
    return transcription
