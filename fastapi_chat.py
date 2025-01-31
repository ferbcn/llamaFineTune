import os
import uvicorn
from transformers import pipeline
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from starlette.staticfiles import StaticFiles
from typing import Dict
import json


load_dotenv()

TOKEN = os.getenv('ACCESS_TOKEN')
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


# model_name = "HuggingFaceH4/zephyr-7b-beta"
# model_name = "mistralai/Mistral-7B-Instruct-v0.3"
# model_name = "fine-tuned-model"
model_name = "meta-llama/Llama-3.2-1B-Instruct"


def generate_text(prompt):
    pipe = pipeline("text-generation", model_name, device="cuda", token=TOKEN)
    messages = [
        {
            "role": "system",
            "content": "You are a friendly chatbot that gives good answers",

        },
        {"role": "user", "content": prompt},
    ]
    response = pipe(messages, max_new_tokens=256)[0]['generated_text'][-1]  # Print the assistant's response
    print(response)
    return response


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Store active websocket connections
active_connections: Dict[str, WebSocket] = {}


@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/chat.html") as f:
        return f.read()


@app.websocket("/ws-chat")
async def websocket_endpoint(websocket: WebSocket):
    # Generate a unique ID for this connection
    connection_id = str(id(websocket))

    try:
        await websocket.accept()

        active_connections[connection_id] = websocket

        # Log connection for debugging
        print(f"Client connected: {connection_id}")

        while True:
            # Wait for messages from the client
            data = await websocket.receive_text()

            # Parse the input message
            message = json.loads(data)
            user_input = message.get("message", "")

            # Send initial message
            await websocket.send_text(json.dumps({
                "type": "start",
                "content": f"<div>Processing: {user_input}</div>"
            }))

            # # Stream the response
            # async for word in generate_text(user_input):
            #     await websocket.send_text(json.dumps({
            #         "type": "chunk",
            #         "content": word
            #     }))

            # return response
            for word in generate_text(user_input)['content']:
                await websocket.send_text(json.dumps({
                    "type": "chunk",
                    "content": word
                }))

            # Send completion message
            await websocket.send_text(json.dumps({
                "type": "end",
                "content": ""
            }))

    except WebSocketDisconnect:
        print(f"Client disconnected: {connection_id}")
    finally:
        # Clean up connection
        if connection_id in active_connections:
            del active_connections[connection_id]




if __name__ == '__main__':
    uvicorn.run(
        'fastapi_chat:app', port=8000, host='0.0.0.0',
        # reload=True,
    )
