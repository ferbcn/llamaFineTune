import os
from transformers import pipeline
from dotenv import load_dotenv
import sys
import torch

load_dotenv()

TOKEN = os.getenv('ACCESS_TOKEN')
# Enable CUDA launch blocking for better error tracking
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


# model_name = "HuggingFaceH4/zephyr-7b-beta"
# model_name = "mistralai/Mistral-7B-Instruct-v0.3"
# model_name = "meta-llama/Llama-3.2-1B-Instruct"
model_name = "fine-tuned-model"


def generate_text(prompt):
    try:
        # Clear CUDA cache before loading model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        pipe = pipeline(
            "text-generation",
            model_name,
            device="cuda" if torch.cuda.is_available() else "cpu",
            token=TOKEN
        )
        
        # Simplify the input format - some fine-tuned models expect plain text
        # rather than the chat format
        try:
            # First try with chat format
            messages = [
                {
                    "role": "system",
                    "content": "You are a friendly chatbot who always responds in the style of scientist",
                },
                {"role": "user", "content": prompt},
            ]
            output = pipe(messages, max_new_tokens=128, do_sample=True)
        except Exception as chat_error:
            print(f"Chat format failed, trying plain text format: {chat_error}")
            # Fallback to plain text format
            system_prompt = "You are a friendly chatbot who always responds in the style of scientist. "
            full_prompt = system_prompt + prompt
            output = pipe(full_prompt, max_new_tokens=128, do_sample=True)

        print(output[0]['generated_text'])

    except Exception as e:
        print(f"Error generating text: {str(e)}")
        print("Try reducing the input size or checking if the model files are correctly loaded")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    command_line = sys.argv[1:]
    prompt = " ".join(command_line)
    if len(prompt) < 5:
        prompt = input("Please enter a prompt: ")
    generate_text(prompt)
