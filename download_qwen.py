import transformers

model_name = "Qwen/Qwen2.5-3B-Instruct"

output_path = "./models/Qwen2.5-3B-Instruct"  # Change this to your desired path

# Download and save the model
model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
model.save_pretrained(output_path, use_safetensors=True)

# Download and save the tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(output_path)
