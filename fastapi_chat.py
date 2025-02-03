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

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# model_name = "HuggingFaceH4/zephyr-7b-beta"
# model_name = "mistralai/Mistral-7B-Instruct-v0.3"
# model_name = "fine-tuned-model"
# model_name = "meta-llama/Llama-3.2-1B-Instruct"

models = {
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


def generate_text_fine_tune(prompt, model_name):
    pipe = pipeline("text-generation", model_name, device="cpu", token=TOKEN)
    messages = [
        {
            "role": "system",
            "content": "You are a friendly chatbot that gives answers",
        },
        {"role": "user", "content": prompt},
    ]
    return str(pipe(messages, max_new_tokens=128)[0]['generated_text'][-1]['content'])


def stream_text(prompt, model_name, max_tokens):
    print("Streaming text for model:", model_name, "with max_tokens:", max_tokens)

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

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

    inputs = tok([formatted_prompt], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    streamer = TextIteratorStreamer(tok, skip_prompt=True)
    generation_kwargs = dict(
        inputs, 
        streamer=streamer, 
        max_new_tokens=max_tokens,
        temperature=0.7,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id
    )
    
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    full_response = ""
    big_chunk = ""

    for new_token in streamer:
        print(new_token, end="")
        # For fine-tuned model, look for the end of the assistant's response
        # Check for various end tokens and clean up the text
        if model_name == "fine-tuned-model":
            if "<|eot_id|>" in new_token:
                # Clean up the text by removing end tokens and any repeated content
                clean_token = new_token.split("<|eot_id|>")[0].strip()
                big_chunk += clean_token
                yield big_chunk
                break

        elif tok.eos_token_id in tok(new_token)['input_ids']:
            clean_token = new_token.split("<|eot_id|>")[0].strip()
            big_chunk += clean_token
            yield big_chunk
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
    max_tokens = data.get("tokens", "")
    # stream_response = generate_text_fine_tuned(user_input, selected_model)
    stream_response = stream_text(user_input, selected_model, max_tokens)
    return StreamingResponse(stream_response, media_type="text/plain")


if __name__ == '__main__':
    uvicorn.run(
        'fastapi_chat:app', port=8000, host='0.0.0.0',
        # reload=True,
    )
