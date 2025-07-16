# Chat AI

This tool objective is to learn about LLMs and the different options for using it.

In this code you:

1. Use your desired LLM model 
2. Write your prompt or talk to the microphone and let an AI model translate it to text.

![model_example](static/Screenshot.png)

It wraps all models so that you can use them seemingly, although different in nature.
The options are:

1. Use an API (ChatGPT, Claude and/or Gemini)
2. Use OLLAMA (install and run locally)
3. Hugging Face pipeline (not implemented)
4. Hugging Face torch model (download python model and run locally)

> [!ATTENTION]
> The voice reader will only work with OpenAI token

## Getting started

Now, to use it, first clone the repository.

Now depending on what you want:

### Hugging Face or API calls

1. Go to the LLM you want to use and create the token.
2. For using the API's, create a `.env` file in the root of the repository.
3. Paste the token as follows (according to your API, you can do them all!)

```
OPENAI_API_KEY=sk-proj-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...
HF_TOKEN=hf_
```

### OLLAMA

The easiest, go to [OLLAMA website](https://ollama.com/) and install in windows.

TODO: You need to run ollama first, my code already downloads the model if needed. But you need to get it running, I need to check that.

### Running streamlit

Run the app with `streamlit run streamlit_app.py`


