import os
from transformers import pipeline
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv('ACCESS_TOKEN')

# model_name="HuggingFaceH4/zephyr-7b-beta"
# model_name= "mistralai/Mistral-7B-Instruct-v0.3"
model_name="meta-llama/Llama-3.2-1B-Instruct"


def generate_text():
    pipe = pipeline("text-generation", model_name, device="cuda", token=TOKEN)
    messages = [
        {
            "role": "system",
            "content": "You are a friendly chatbot who always responds in the style of a pirate",
        },
        {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
    ]
    print(pipe(messages, max_new_tokens=128)[0]['generated_text'][-1])  # Print the assistant's response


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    generate_text()
