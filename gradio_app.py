import gradio as gr
from models import get_available_models, get_model

# Initialize session state
chat_history = []
model_selector = {"model": None, "name": None}

def init_model(model_name):
    if model_selector["name"] != model_name:
        model_selector["model"] = get_model(model_name)
        model_selector["name"] = model_name
        chat_history.clear()
    return f"Model '{model_name}' loaded."

def chat(prompt, model_name):
    if not prompt.strip():
        return "", chat_history

    init_model(model_name)
    model = model_selector["model"]
    model.messages.append({"role": "user", "content": prompt})
    chat_history.append(("You", prompt))

    response = ""
    for chunk in model.chat(prompt):
        response = chunk
        yield response, chat_history + [("AI", response)]

    model.messages.append({"role": "assistant", "content": response})

# UI
with gr.Blocks(title="Chat AI") as demo:
    gr.Markdown("## 🤖 Chat AI Interface")

    model_list = get_available_models()
    with gr.Row():
        model_dropdown = gr.Dropdown(choices=model_list, label="Select Model", value=model_list[0])

    chatbot = gr.Chatbot(label="Chat", type="messages")
    prompt_box = gr.Textbox(lines=4, placeholder="Enter your prompt here...", label="Prompt")

    send_button = gr.Button("Send")

    # Streaming output function
    def on_send(prompt, model_name):
        return chat(prompt, model_name)

    send_button.click(
        fn=on_send,
        inputs=[prompt_box, model_dropdown],
        outputs=[chatbot, chatbot],
    )

if __name__ == "__main__":
    demo.launch()
