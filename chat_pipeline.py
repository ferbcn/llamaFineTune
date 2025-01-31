import os
from transformers import pipeline
from dotenv import load_dotenv
import sys

load_dotenv()

TOKEN = os.getenv('ACCESS_TOKEN')
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


# model_name = "HuggingFaceH4/zephyr-7b-beta"
# model_name = "mistralai/Mistral-7B-Instruct-v0.3"
# model_name = "meta-llama/Llama-3.2-1B-Instruct"
model_name = "fine-tuned-model"


def generate_text(prompt):
    pipe = pipeline("text-generation", model_name, device="cuda", token=TOKEN)
    messages = [
        {
            "role": "system",
#            "content": "You are a friendly chatbot who always responds in the style of a pirate",
            "content": "You are a friendly chatbot who always responds in the style of scientist",

        },
        {"role": "user", "content": prompt},
    ]
    print(pipe(messages, max_new_tokens=512)[0]['generated_text'][-1])  # Print the assistant's response


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    command_line = sys.argv[1:]
    prompt = " ".join(command_line)
    if len(prompt) < 5:
        prompt = input("Please enter a prompt:")
    generate_text(prompt)
