from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import json

model_dir = "model/model_finetuned"  # or wherever you saved your model
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir).to(device)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if device == "cuda" else -1)

# Example prompt
prompt = "I want to add a cheeseburger and fries to my basket."

output = pipe(prompt, max_new_tokens=100, do_sample=False)[0]
print(f"🔁 Response:\n\n{json.dumps(output, indent=4)}")
