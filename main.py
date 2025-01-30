from transformers import pipeline
from transformers import AutoTokenizer


def generate_text():
    # # Using an open-source model instead of gated Llama model
    # model_id = "facebook/opt-350m"
    # pipe = pipeline(
    #     "text-generation",
    #     model=model_id,
    #     device_map="auto",
    # )
    # messages = [
    #     {"role": "system", "content": "You are a helpful assistant."},
    #     {"role": "user", "content": "Should I let my dog sleep on the bed with me?"},
    # ]
    # tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
    # tokenizer.apply_chat_template(messages, tokenize=False)
    #
    # outputs = pipe(
    #     tokenizer,
    #     max_new_tokens=128,
    #     do_sample=True
    # )
    # print(outputs[0]["generated_text"])
    pipe = pipeline("text-generation", "HuggingFaceH4/zephyr-7b-beta", device="gpu")
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
