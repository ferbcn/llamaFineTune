import asyncio
import os
from threading import Thread

import torch
import uvicorn
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from starlette.staticfiles import StaticFiles
import json

load_dotenv()

TOKEN = os.getenv('ACCESS_TOKEN')

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# model_name = "HuggingFaceH4/zephyr-7b-beta"
# model_name = "mistralai/Mistral-7B-Instruct-v0.3"
# model_name = "fine-tuned-model"
model_name = "meta-llama/Llama-3.2-1B-Instruct"

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


def stream_text(prompt):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Get the EOT token ID if it exists in the tokenizer
    eot_token_id = tok.eos_token_id

    # Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    inputs = tok([prompt], return_tensors="pt")
    # Move all input tensors to the same device as the model
    inputs = {k: v.to(device) for k, v in inputs.items()}

    streamer = TextIteratorStreamer(tok, skip_prompt=True)
    generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=512)
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    big_chunk = ""
    for new_text in streamer:
        # Check if the current token is the EOT token
        if any(token == eot_token_id for token in tok(new_text)['input_ids']):
            print("EOT token detected!")
            if big_chunk:  # Yield any remaining text before EOT
                yield big_chunk
            break

        # add tokens to big_chunk
        print(new_text, end="")
        big_chunk += new_text
        if len(big_chunk) > 100:
            yield big_chunk
            big_chunk = ""


@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/chat.html") as f:
        return f.read()


@app.post("/stream")
async def generate(request: Request):
    data = await request.json()
    user_input = data.get("message", "")
    stream_response = stream_text(user_input)
    return StreamingResponse(stream_response, media_type="text/plain")


if __name__ == '__main__':
    uvicorn.run(
        'fastapi_chat:app', port=8000, host='0.0.0.0',
        # reload=True,
    )
