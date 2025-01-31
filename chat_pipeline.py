import os
from transformers import AutoModelForCausalLM, AutoTokenizer
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Load model and tokenizer with the same configuration as fine-tuning
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Match the fine-tuning dtype
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        # Prepare the input in the same format as training data
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        # Apply the same chat template as used in training
        full_prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        # Tokenize input
        inputs = tokenizer(
            full_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128  # Same as training
        )

        # Move inputs to the same device as model
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            no_repeat_ngram_size=2,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        # Decode and print the response
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(generated_text)

    except Exception as e:
        print(f"Error: {str(e)}")
        print("Try reducing the input size or checking if the model files are correctly loaded")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    command_line = sys.argv[1:]
    prompt = " ".join(command_line)
    if len(prompt) < 5:
        prompt = input("Please enter a prompt: ")
    generate_text(prompt)
