import os
from transformers import pipeline
from dotenv import load_dotenv
import sys

load_dotenv()

TOKEN = os.getenv('ACCESS_TOKEN')

model_name = "fine-tuned-model"


def generate_text(prompt):
    pipe = pipeline("text-generation", model_name, device="cuda", token=TOKEN)
    messages = [
        {
            "role": "system",
            "content": "You are a friendly chatbot that gives answers",

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
