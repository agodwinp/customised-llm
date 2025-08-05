from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from pathlib import Path
import json

# Paths
jsonl_path = "data/dataset.jsonl"
output_path = "data/flattened_dataset"
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # For padding compatibility

# Load raw dataset
raw_ds = load_dataset("json", data_files=jsonl_path, split="train")

# Flatten to instruction → output pairs
def flatten_conversation(example):
    system_prompt = """## ROLE
You are a helpful AI-powered drive-through ordering assistant.

## CONTEXT
You help users order food or drinks via voice at a drive-through.

## TASK
1. Understand the last 3 messages of the conversation
2. Update the basket as needed
3. Return a structured JSON response containing:
   - intent
   - updated basket
   - spoken response string

## RULES
Only return valid JSON that conforms to the schema.
Do not add extra commentary outside the JSON.

## DATA
"""
    # Include only the last 3 messages (if more, truncate from start)
    message_context = example["messages"][-3:]
    history = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in message_context])
    prompt = f"{system_prompt}{history}"
    output = json.dumps(example["output"], ensure_ascii=False)
    return {"instruction": prompt, "output": output}

flattened_ds = raw_ds.map(flatten_conversation, remove_columns=raw_ds.column_names)

# Tokenize the data
def tokenize(example):
    merged = example["instruction"] + "\n" + example["output"]
    return tokenizer(merged, truncation=True, padding="max_length", max_length=512)

tokenized_ds = flattened_ds.map(tokenize)

# Train/test split
ds_split = tokenized_ds.train_test_split(test_size=0.1, seed=42)

# Save to disk
Path(output_path).mkdir(parents=True, exist_ok=True)
ds_split.save_to_disk(output_path)

print(f"✅ Saved tokenized dataset to `{output_path}`")
