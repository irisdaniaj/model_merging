import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score
import json

# Define model name and paths
model_name = 'huawei-noah/TinyBERT_General_4L_312D'  # TinyBERT model identifier
prepared_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "prepared")
models_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "tiny-bert")

# Ensure the models directory exists
os.makedirs(models_path, exist_ok=True)

# Define datasets to fine-tune on
datasets_to_finetune = ['stsb', 'sst2', 'rte']

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define function to compute metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return {"accuracy": acc, "f1": f1}

# Tokenization function
def tokenize_function(example, dataset_name):
    if dataset_name == 'stsb':
        # Tokenization for sentence pair tasks (STS-B)
        return tokenizer(example['sentence1'], example['sentence2'], padding="max_length", truncation=True)
    elif dataset_name == 'rte':
        # Tokenization for sentence pair tasks (RTE)
        return tokenizer(example['sentence1'], example['sentence2'], padding="max_length", truncation=True)
    else:
        # Tokenization for single sentence tasks (SST-2)
        return tokenizer(example['sentence'], padding="max_length", truncation=True)

# Fine-tune TinyBERT on each dataset
for dataset_name in datasets_to_finetune:
    print(f"Fine-tuning TinyBERT on {dataset_name.upper()} dataset...")

    # Load prepared dataset from disk
    dataset = load_from_disk(os.path.join(prepared_data_path, dataset_name))

    # Determine the number of labels
    num_labels = 1 if dataset_name == 'stsb' else 2  # 1 for regression (STS-B), 2 for classification (SST-2, RTE)

    # Load TinyBERT model from pre-trained weights
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    # Tokenize the dataset
    tokenized_datasets = dataset.map(lambda x: tokenize_function(x, dataset_name), batched=True)

    # Define training arguments for fine-tuning
    training_args = TrainingArguments(
        output_dir=os.path.join(models_path, f"{dataset_name}_finetuned"),
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,  # Adjust epochs as necessary for fine-tuning
        weight_decay=0.01,
        save_strategy="epoch",
        logging_dir='./logs',
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy"
    )

    # Initialize Trainer for fine-tuning
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Fine-tune the model
    trainer.train()

    # Save the fine-tuned model and tokenizer
    model_save_path = os.path.join(models_path, f"{dataset_name}_finetuned")
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)

    # Save the training arguments (hyperparameters)
    hyperparameters_save_path = os.path.join(model_save_path, "training_args.json")
    with open(hyperparameters_save_path, "w") as f:
        json.dump(training_args.to_dict(), f)

    # Save the training metrics
    metrics = trainer.evaluate()
    metrics_save_path = os.path.join(model_save_path, "metrics.json")
    with open(metrics_save_path, "w") as f:
        json.dump(metrics, f)

    print(f"Model fine-tuned on {dataset_name.upper()} and saved in {model_save_path}")
    print(f"Metrics saved in {metrics_save_path}")
    print(f"Hyperparameters saved in {hyperparameters_save_path}")

print("Fine-tuning complete for all datasets!")
# The script above demonstrates how to fine-tune a TinyBERT model on multiple datasets using the Hugging Face Transformers library. The script loads a pre-trained TinyBERT model, tokenizes the datasets, defines training arguments, and fine-tunes the model on each dataset. It also saves the fine-tuned model, tokenizer, training metrics, and hyperparameters for each dataset. The script can be customized to fine-tune on additional datasets or adjust hyperparameters as needed.