import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
import evaluate

# 1. Load Dataset from CSV
dataset_file = "customer_data.csv"
print(f"Loading dataset from: {dataset_file}")
dataset = load_dataset("csv", data_files=dataset_file)

# Split into train/test (since we only have one file)
dataset = dataset["train"].train_test_split(test_size=0.2)

# 2. Preprocess Data
model_ckpt = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

# Define label mappings
id2label = {
    0: "purchase_intent",
    1: "product_details",
    2: "general_praise",
    3: "shipping_inquiry",
    4: "complaint"
}
label2id = {v: k for k, v in id2label.items()}

def preprocess_function(examples):
    # Tokenize text
    tokenized = tokenizer(examples["text"], padding=True, truncation=True)    
    label_ids = []
    for label in examples["label"]:
        if label in label2id:
            label_ids.append(label2id[label])
        else:
            # Fallback or error
            label_ids.append(0) 
            
    tokenized["label"] = label_ids
    return tokenized

print("Tokenizing and mapping labels...")
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# 3. Load Metrics
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy.compute(predictions=predictions, references=labels)
    f1_score = f1.compute(predictions=predictions, references=labels, average="weighted")
    return {"accuracy": acc["accuracy"], "f1": f1_score["f1"]}

# 4. Initialize Model
num_labels = len(id2label)
print(f"Initializing model: {model_ckpt} with {num_labels} labels")
model = AutoModelForSequenceClassification.from_pretrained(
    model_ckpt, num_labels=num_labels, id2label=id2label, label2id=label2id
)

# 5. Define Training Arguments
batch_size = 4
logging_steps = 1
output_dir = "./results"

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=4,
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
    logging_steps=logging_steps,
    report_to="wandb"
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 6. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 7. Train
print("Starting training...")
trainer.train()

# 8. Evaluate
print("Evaluating on test set...")
test_results = trainer.evaluate(tokenized_dataset["test"])
print(f"Test Results: {test_results}")

# 9. Save Model
save_path = "./emotion_bert_model" # Keep same path for inference.py compatibility
print(f"Saving model to {save_path}")
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)
print("Done!")
