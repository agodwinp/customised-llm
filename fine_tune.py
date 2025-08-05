# from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, TrainerCallback
# from datasets import load_from_disk
# from peft import get_peft_model, LoraConfig, TaskType
# import torch

# class DebugCallback(TrainerCallback):
#     def on_log(self, args, state, control, logs=None, **kwargs):
#         print("\nLOG:", logs)

# # Load dataset
# train_test = load_from_disk("data/flattened_dataset")

# # Tokenizer + model
# model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.pad_token = tokenizer.eos_token  # Required if model lacks a pad token

# # Determine precision
# use_fp16 = torch.cuda.is_available()

# # Tokenization function
# def tokenize_fn(batch):
#     prompts = [f"{i}\n{o}" for i, o in zip(batch["instruction"], batch["output"])]
#     tokens = tokenizer(
#         prompts,
#         truncation=True,
#         padding="max_length",
#         max_length=512,
#     )

#     # Mask padding in labels
#     labels = []
#     for input_ids in tokens["input_ids"]:
#         label = [token if token != tokenizer.pad_token_id else -100 for token in input_ids]
#         labels.append(label)

#     tokens["labels"] = labels
#     return tokens

# # Tokenize dataset
# tokenized = train_test.map(tokenize_fn, batched=True)
# tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
# print(tokenized["train"][0].keys())

# # Load base model
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     use_safetensors=True,
#     torch_dtype=torch.float16 if use_fp16 else torch.float32,
#     device_map=None
# )

# # Apply LoRA
# peft_config = LoraConfig(
#     task_type=TaskType.CAUSAL_LM,
#     r=8,
#     lora_alpha=32,
#     lora_dropout=0.1,
#     bias="none"
# )
# model = get_peft_model(model, peft_config)
# model.print_trainable_parameters()

# # Force model to use half precision if applicable (fixes unscale error)
# # if use_fp16:
# #     model = model.half()

# # Training arguments
# training_args = TrainingArguments(
#     output_dir="./model/model_lora",
#     per_device_train_batch_size=2,
#     gradient_accumulation_steps=8,
#     logging_dir="./model/logs",
#     logging_steps=50,
#     eval_strategy="steps",
#     eval_steps=100,
#     save_strategy="steps",
#     save_steps=100,
#     num_train_epochs=2,
#     learning_rate=2e-4,
#     warmup_steps=100,
#     weight_decay=0.01,
#     fp16=False,
#     bf16=False,
#     report_to="none",
# )

# # Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized["train"],
#     eval_dataset=tokenized["test"],
#     callbacks=[DebugCallback()]
# )

# # Train and save
# trainer.train()
# model.save_pretrained("model/model_finetuned")
# tokenizer.save_pretrained("model/model_finetuned")


from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, TrainerCallback
from datasets import load_from_disk
from peft import get_peft_model, LoraConfig, TaskType
import torch

# Debug logging
class DebugCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        print("\n🟨 LOG:", logs)

# Load dataset
train_test = load_from_disk("data/flattened_dataset")

# Load tokenizer
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Ensure padding token is defined

# Tokenize function with -100 masking
def tokenize_fn(batch):
    prompts = [f"{i}\n{o}" for i, o in zip(batch["instruction"], batch["output"])]
    tokens = tokenizer(
        prompts,
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    labels = []
    for input_ids in tokens["input_ids"]:
        label = [
            tok if tok != tokenizer.pad_token_id else -100
            for tok in input_ids
        ]
        labels.append(label)
    tokens["labels"] = labels
    return tokens

# Tokenize and format
tokenized = train_test.map(tokenize_fn, batched=True)
tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

print("\n✅ Dataset columns:", tokenized["train"].column_names)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    use_safetensors=True,
    torch_dtype=torch.float32,  # Force FP32 for stable debug
    device_map="auto"
)

# Apply LoRA
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none"
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Sanity check: confirm trainable and unfrozen
print("\n✅ LoRA trainable params:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print("🔓", name)
    else:
        print("🚫", name)

# Manual forward pass test on one sample
print("\n🧪 Running forward+backward on single sample...")
sample = tokenized["train"][0]
for k in sample:
    sample[k] = sample[k].unsqueeze(0)  # Add batch dimension

sample = {k: v.to(model.device) for k, v in sample.items()}
model.train()
outputs = model(**sample)
print("✅ Initial loss:", outputs.loss.item())

outputs.loss.backward()
print("✅ .backward() successful — gradients should flow.\n")

# Training args
training_args = TrainingArguments(
    output_dir="./model/model_lora",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    logging_dir="./model/logs",
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=100,
    num_train_epochs=2,
    learning_rate=2e-4,
    warmup_steps=100,
    weight_decay=0.01,
    fp16=True,  # Leave off for stability
    bf16=False,
    report_to="none",
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    callbacks=[DebugCallback()]
)

# Train and save
trainer.train()
model.save_pretrained("model/model_finetuned")
tokenizer.save_pretrained("model/model_finetuned")
