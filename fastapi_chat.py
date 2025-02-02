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


async def stream_text(prompt):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    inputs = tok([prompt], return_tensors="pt")

    if torch.cuda.is_available():
        inputs["input_ids"] = inputs["input_ids"].to("cuda")
    else:
        inputs["input_ids"] = inputs["input_ids"]

    streamer = TextIteratorStreamer(tok, skip_prompt=True)
    generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=512)
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for new_text in streamer:
        print(new_text, end="")
        yield new_text
        await asyncio.sleep(0.001)


@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/chat.html") as f:
        return f.read()


@app.post("/stream")
async def stream(request: Request):
    data = await request.json()
    user_input = data.get("message", "")

    async def event_generator():
        yield "data: " + json.dumps({
            "type": "start",
            "content": f"Processing: {user_input}"
        }) + "\n\n"
        
        async for token in stream_text(user_input):
            yield "data: " + json.dumps({
                "type": "chunk",
                "content": token
            }) + "\n\n"

        yield "data: " + json.dumps({
            "type": "end",
            "content": ""
        }) + "\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'  # Disable buffering in Nginx
        }
    )


if __name__ == '__main__':
    uvicorn.run(
        'fastapi_chat:app', port=8000, host='0.0.0.0',
        # reload=True,
    )
