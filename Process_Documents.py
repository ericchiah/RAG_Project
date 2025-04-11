from datasets import Dataset

# === CONFIGURATION ===
MODEL_NAME = "EleutherAI/gpt-neo-125M"  # Small model for testing
DATA_FILE = "training_data.jsonl"  # Your prompt-completion file
MAX_LEN = 512
OUTPUT_DIR = "./finetuned_rescheduler_model"


# === 1. Load JSONL and Format ===
def load_jsonl_data(path):
    import json
    from datasets import Dataset
    with open(path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    # Convert to "text" field format for training
    return [{"text": f"### Prompt:\n{item['prompt']}\n\n### Response:\n{item['completion']}"} for item in data]


raw_data = load_jsonl_data(DATA_FILE)
dataset = Dataset.from_list(raw_data)

# === 2. Load Model and Tokenizer ===
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Set pad_token to eos_token (if available)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Use eos_token as pad_token


# === 3. Tokenize Dataset ===
def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=MAX_LEN)


tokenized_dataset = dataset.map(tokenize, batched=True)

# === 4. Training Arguments ===
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    fp16=False,  # Change to True if on supported GPU
)

# === 5. Start Training ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

trainer.train()

# === 6. Save Model Locally ===
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ Fine-tuning complete. Model saved at:", OUTPUT_DIR)
