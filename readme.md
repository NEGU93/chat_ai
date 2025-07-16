# 🤖 Chat AI - Universal LLM Interface

> A production-ready, multi-provider AI chat application with voice integration and unified API abstraction

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)

## 🎯 Overview

**Chat AI** is a sophisticated conversational AI platform that abstracts away the complexity of working with multiple Large Language Models (LLMs). Built with modern Python architecture, it provides a unified interface for interacting with various AI providers through a single, elegant web application.

![UI](static/Screenshot.png)

### 🔥 Key Features

- **🔄 Multi-Provider Support**: Seamlessly switch between OpenAI GPT, Claude, Gemini, LLAMA, and other models
- **🎙️ Voice Integration**: Speech-to-text capabilities with real-time transcription
- **🏠 Local & Cloud**: Support for both cloud APIs and local inference (OLLAMA, Hugging Face)
- **🎨 Modern UI**: Clean, responsive interface built with Streamlit
- **🔧 Extensible Architecture**: Modular design allowing easy addition of new AI providers
- **⚡ Performance Optimized**: Efficient model loading and caching mechanisms

### 🧠 Technical Highlights

- **Provider Abstraction Layer**: Unified interface for different AI models regardless of their underlying implementation
- **Async Processing**: Non-blocking operations for better user experience
- **Configuration Management**: Secure environment-based configuration
- **Scalable Design**: Easy to extend with new AI providers

## 🚀 Supported AI Providers

| Provider | Type | Default Model |
|----------|------|----------|
| **OpenAI** | Cloud API | gpt-4o-mini |
| **Claude** | Cloud API | claude-3-7-sonnet-latest |
| **Gemini** | Cloud API | gemini-2.0-flash |
| **OLLAMA** | Local | llama3.2 |
| **Hugging Face** | Local | Meta-Llama-3.1-8B-Instruct |

## 🛠️ Installation & Setup

### Quick Start

```bash
# Clone the repository
git clone https://github.com/NEGU93/chat_ai.git
cd chat_ai

# Install dependencies
pip install -r requirements.txt

# Edit .env with your API keys
```

### Setup keys

Setup whatever keys you want

- For OpenAI, visit https://openai.com/api/
- For Anthropic, visit https://console.anthropic.com/
- For Google, visit https://ai.google.dev/gemini-api
- For HuggingFace, visit https://huggingface.co

> [!ATTENTION]
> OpenAI key is required for Speech-to-text capabilities

> [!ATTENTION]
> For the default HuggingFace model, Meta requires you to sign their terms of service. Visit their model instructions page in Hugging Face: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B. It might take some time to validate.

### Environment Configuration

Create a `.env` file in the root directory:

```env
# Cloud API Keys (use only what you need)
OPENAI_API_KEY=sk-proj-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...
HF_TOKEN=hf_...
```

### Local Setup (OLLAMA)

For local AI models without API costs:

1. Install [OLLAMA](https://ollama.com/)
2. Run the application - models will be downloaded automatically but will take a long time the first run
3. Enjoy unlimited local AI chat

## 🎮 Usage

```bash
# Start the application
streamlit run streamlit_app.py
```

Navigate to `http://localhost:8501` and start chatting with your preferred AI model!

## 🎤 Voice Features

The integrated voice system allows for:
- **Real-time transcription** of spoken input
- **Hands-free interaction** with AI models
- **Multi-language support** (via OpenAI Whisper)

*Note: Voice features require OpenAI API key*

## 💡 Use Cases

- **AI Research & Development**: Compare different models' responses
- **Education**: Learn about various LLM capabilities
- **Prototyping**: Quickly test AI-powered features
- **Cost Optimization**: Mix cloud and local models based on needs
- **Privacy**: Use local models for sensitive conversations

## 🔧 Technical Implementation

### Core Components

- **Model Abstraction**: `models.py` implements a unified interface pattern
- **UI Layer**: `streamlit_app.py` provides responsive web interface
- **Configuration**: Environment-based secure API key management
- **Error Handling**: Comprehensive exception handling across all providers

### Performance Optimizations

- Lazy loading of AI models
- Efficient memory management
- Caching for improved response times
- Async operations where applicable

## 📊 Why This Project Matters

This application showcases several important technical skills:

- **AI/ML Integration**: Practical experience with multiple LLM providers
- **Software Architecture**: Clean, modular design patterns
- **Full-Stack Development**: Backend AI logic + Frontend web interface
- **API Management**: Secure handling of multiple API integrations
- **Performance Engineering**: Optimization for real-time AI interactions


## 🏗️ Architecture

The application follows a clean architecture pattern with clear separation of concerns:

```
├── models.py          # Core AI provider abstractions and unified interface
├── streamlit_app.py   # Web application frontend and user interaction
├── requirements.txt   # Python dependencies
└── .env              # Environment configuration (API keys)
```

---

**Built with ❤️ by NEGU93** | [GitHub](https://github.com/NEGU93) | [Connect on LinkedIn](https://linkedin.com/in/your-profile)

*Showcasing modern AI application development with production-ready architecture*