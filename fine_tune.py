from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_from_disk
from peft import get_peft_model, LoraConfig, TaskType
import torch

# Load dataset
train_test = load_from_disk("model/flattened_dataset")

# Tokenizer + model
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Required if model lacks a pad token

# Tokenize function
def tokenize_fn(example):
    prompt = f"{example['instruction']}\n{example['output']}"
    tokens = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )
    tokens = {k: v.squeeze() for k, v in tokens.items()}
    tokens["labels"] = tokens["input_ids"].clone()
    return tokens

tokenized = train_test.map(tokenize_fn, batched=False)

# Load model (no quantization)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    use_safetensors=True,
    device_map=None,
    torch_dtype=torch.float32,
)

# LoRA config
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none"
)
model = get_peft_model(model, peft_config)

# Training args
training_args = TrainingArguments(
    output_dir="./model/model_lora",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    logging_dir="./logs",
    logging_steps=50,
    # evaluation_strategy="steps",
    eval_strategy="steps",
    save_strategy="steps",
    save_steps=200,
    eval_steps=200,
    num_train_epochs=3,
    learning_rate=2e-4,
    warmup_steps=100,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),
    report_to="none",
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    # fp16=False,
    # bf16=False,
)

# Train + Save
trainer.train()
model.save_pretrained("model/model_finetuned")
tokenizer.save_pretrained("model/model_finetuned")
