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

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# model_name = "HuggingFaceH4/zephyr-7b-beta"
# model_name = "mistralai/Mistral-7B-Instruct-v0.3"
# model_name = "fine-tuned-model"
# model_name = "meta-llama/Llama-3.2-1B-Instruct"

models = {
    "mistral7B": "mistralai/Mistral-7B-Instruct-v0.3",
    "llama3.2": "meta-llama/Llama-3.2-1B-Instruct",
    "llama3.2-fine-tuned": "fine-tuned-model"  # This should point to your local model directory
}

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/chat.html") as f:
        return f.read()


@app.get("/get-models")
async def get_models():
    return models


def generate_text(prompt, model_name):
    pipe = pipeline("text-generation", model_name, device="cpu", token=TOKEN)
    messages = [
        {
            "role": "system",
            "content": "You are a friendly chatbot that gives answers",
        },
        {"role": "user", "content": prompt},
    ]
    return str(pipe(messages, max_new_tokens=128)[0]['generated_text'][-1]['content'])


def stream_text(prompt, model_name):
    print("Streaming text for model:", model_name)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Get the EOT token ID if it exists in the tokenizer
    eot_token_id = tok.eos_token_id

    # Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    if model_name == "fine-tuned-model":
        # Format prompt similar to how it was formatted during fine-tuning
        formatted_prompt = f"### Human: {prompt}\n\n### Assistant:"
    else:
        formatted_prompt = prompt

    inputs = tok([formatted_prompt], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    streamer = TextIteratorStreamer(tok, skip_prompt=True)
    generation_kwargs = dict(
        inputs, 
        streamer=streamer, 
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id
    )
    
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    big_chunk = ""
    for new_text in streamer:
        # For fine-tuned model, check for end of response markers
        if model_name == "fine-tuned-model":
            if "### Human:" in new_text or "### Assistant:" in new_text:
                if big_chunk:
                    yield big_chunk
                break
        # For other models, check EOT token
        elif any(token == eot_token_id for token in tok(new_text)['input_ids']):
            if big_chunk:
                yield big_chunk
            break

        print(new_text, end="")
        big_chunk += new_text
        if len(big_chunk) > 200 and "\n" in new_text:
            yield big_chunk
            big_chunk = ""
    
    # Yield any remaining text
    if big_chunk:
        yield big_chunk


@app.post("/stream")
async def generate(request: Request):
    data = await request.json()
    user_input = data.get("message", "")
    selected_model = data.get("model", "")  # Use default if not specified
    # stream_response = stream_text(user_input, selected_model)
    stream_response = generate_text(user_input, selected_model)
    return StreamingResponse(stream_response, media_type="text/plain")


if __name__ == '__main__':
    uvicorn.run(
        'fastapi_chat:app', port=8000, host='0.0.0.0',
        # reload=True,
    )
