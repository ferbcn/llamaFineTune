import os
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
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

        # Load model and tokenizer separately for more control
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        if torch.cuda.is_available():
            model = model.to("cuda")

        # Create pipeline with specific parameters to handle probability issues
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device="cuda" if torch.cuda.is_available() else "cpu",
            token=TOKEN,
            temperature=0.7,  # Lower temperature for more stable outputs
            top_p=0.9,       # Nucleus sampling parameter
            top_k=50,        # Top-k sampling parameter
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        
        try:
            # Use plain text format as it's more likely to work with fine-tuned models
            system_prompt = "You are a friendly chatbot who always responds in the style of scientist. "
            full_prompt = system_prompt + prompt
            
            output = pipe(
                full_prompt,
                max_new_tokens=128,
                do_sample=True,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                repetition_penalty=1.2
            )

            print(output[0]['generated_text'])

        except Exception as inner_error:
            print(f"Error during text generation: {str(inner_error)}")

    except Exception as e:
        print(f"Error setting up the model: {str(e)}")
        print("Try reducing the input size or checking if the model files are correctly loaded")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    command_line = sys.argv[1:]
    prompt = " ".join(command_line)
    if len(prompt) < 5:
        prompt = input("Please enter a prompt: ")
    generate_text(prompt)
