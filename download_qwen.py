import transformers

model_name = "Qwen/Qwen2.5-1.5B"

output_path = "./models/Qwen2.5-1.5B"  # Change this to your desired path
model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
model.save_pretrained(output_path, use_safetensors=True)
