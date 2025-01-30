from transformers import pipeline


def generate_text():
    pipe = pipeline("text-generation", "HuggingFaceH4/zephyr-7b-beta", device="cpu")
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
