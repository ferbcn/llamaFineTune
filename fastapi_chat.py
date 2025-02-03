import os
from threading import Thread
import torch
import uvicorn
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer, pipeline
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from starlette.staticfiles import StaticFiles

load_dotenv()

TOKEN = os.getenv('ACCESS_TOKEN')

BIG_CHUNK_SIZE = 200

models = {
    "llama3.2": "meta-llama/Llama-3.2-1B-Instruct",
    "llama3.2-fine-tuned": "fine-tuned-model",  # This should point to your local model directory
    "deepseekR1": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "deepseek7B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseekLlama8B": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "zephyr7B": "HuggingFaceH4/zephyr-7b-beta",
}

inference_options = [
    {"label": "Max Tokens", "label_id": "token-value", "slider_id": "token-slider", "min":16, "max":1024, "value":256, "step":16},
    {"label": "Temperature", "label_id": "temp-value", "slider_id": "temp-slider", "min":0.1, "max":1.0, "value":0.7, "step":0.05},
    {"label": "Top-P", "label_id": "topp-value", "slider_id": "topp-slider", "min":0.9, "max":0.99, "value":0.95, "step":0.01}]

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/chat.html") as f:
        return f.read()


@app.get("/get-models")
async def get_models():
    return models


@app.get("/get-options")
async def get_options():
    return inference_options


def stream_text(prompt, model_name, max_tokens=256, temp=0.7, top_p=0.95):
    print("Streaming text for model:", model_name, "with max_tokens:", max_tokens, "with temp:", temp, "with top_p:", top_p)

    tok = AutoTokenizer.from_pretrained(model_name)
    # Load model with float16 precision
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"  # This will automatically handle device placement
    )

    # Get the device that's actually being used
    device = next(model.parameters()).device
    
    if model_name == "fine-tuned-model":
        # Format prompt using the tokenizer's chat template
        messages = [
            {"role": "user", "content": prompt}
        ]
        formatted_prompt = tok.apply_chat_template(
            messages, 
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        formatted_prompt = prompt

    # Use the device for inputs
    inputs = tok([formatted_prompt], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    streamer = TextIteratorStreamer(tok, skip_prompt=True)
    generation_kwargs = dict(
        inputs, 
        streamer=streamer, 
        max_new_tokens=max_tokens,
        temperature=temp,
        top_p=top_p,
        do_sample=True,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id
    )
    
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    full_response = ""
    big_chunk = ""

    for new_token in streamer:
        print(new_token)
        # For fine-tuned model, look for the end of the assistant's response
        # Check for various end tokens and clean up the text
        if tok.eos_token_id in tok(new_token)['input_ids'] or "<|eot_id|>" in new_token or "<｜end▁of▁sentence｜>" in new_token:
            clean_token = new_token.split("<|eot_id|>")[0].strip().split("<｜end▁of▁sentence｜>>")[0].strip()
            big_chunk += clean_token
            yield big_chunk
            big_chunk = ""
            break

        big_chunk += new_token
        full_response += new_token

        # yielding every token will break formatting at the frontend
        if device == "cpu" or len(full_response) < BIG_CHUNK_SIZE or len(big_chunk) > BIG_CHUNK_SIZE and "\n" in new_token:
            yield big_chunk
            big_chunk = ""

    # Yield any remaining text
    if big_chunk:
        yield big_chunk


@app.post("/stream")
async def generate(request: Request):
    data = await request.json()
    user_input = data.get("message", "")
    selected_model = data.get("model", "")
    max_tokens = data.get("token", "")
    temp = data.get("temp", "")
    top_p = data.get("topp", "")
    # print("Post data:", user_input, selected_model, max_tokens, temp, top_p)
    stream_response = stream_text(user_input, selected_model, max_tokens=max_tokens, temp=temp, top_p=top_p)
    return StreamingResponse(stream_response, media_type="text/plain")


if __name__ == '__main__':
    uvicorn.run(
        'fastapi_chat:app', port=8000, host='0.0.0.0',
        # reload=True,
    )
