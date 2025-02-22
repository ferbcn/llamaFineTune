import os

import torch
from transformers import pipeline
from dotenv import load_dotenv
import sys

load_dotenv()

TOKEN = os.getenv('ACCESS_TOKEN')


# Define the available model names
model_names = {
    "1": "fine-tuned-model",
    "2": "meta-llama/Llama-3.2-1B-Instruct",
    "3": "mistralai/Mistral-7B-Instruct-v0.3",
    "4": "HuggingFaceH4/zephyr-7b-beta",
    "5": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "6": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "7": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
}

def generate_text(prompt, model_name):
    if torch.cuda.is_available():
        device = "cuda"
        # Use FP16 precision for larger models
        torch_dtype = torch.float16
    else:
        device = "cpu"
        torch_dtype = torch.float32
        
    pipe = pipeline(
        "text-generation",
        model_name,
        device=device,
        token=TOKEN,
        torch_dtype=torch_dtype
    )
    
    messages = [
        {
            "role": "system",
            "content": "You reason and explain like good teacher",
        },
        {"role": "user", "content": prompt},
    ]
    print(pipe(messages, max_new_tokens=512)[0]['generated_text'][-1])  # Print the assistant's response

if __name__ == '__main__':
    print("Select a model:")
    for key, value in model_names.items():
        print(f"{key}: {value}")
    
    selected_model = input("Enter the number of the model you want to use: ")
    
    # Validate the selected model
    if selected_model not in model_names:
        print("Invalid model selection.")
        sys.exit(1)
    
    prompt = input("Please enter a prompt: ")
    
    generate_text(prompt, model_names[selected_model])