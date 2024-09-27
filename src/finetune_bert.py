import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
import scipy.stats
import json

# Define model name and paths
# Define model name and paths for BERT base instead of TinyBERT
model_name = 'bert-base-uncased'  # BERT base model identifier
prepared_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "prepared")
models_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "bert-base")

# Ensure the models directory exists
os.makedirs(models_path, exist_ok=True)

# Define datasets to fine-tune on
datasets_to_finetune = [ 'sst2', 'rte']

# Load tokenizer for BERT base
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define function to compute metrics
def compute_metrics(eval_pred, dataset_name):
    predictions, labels = eval_pred
    preds = predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return {"accuracy": acc, "f1": f1}

# Tokenization function
def tokenize_function(example, dataset_name):
    if dataset_name == 'rte':
        # Tokenization for sentence pair tasks (RTE)
        return tokenizer(example['sentence1'], example['sentence2'], padding="max_length", truncation=True)
    else:
        # Tokenization for single sentence tasks (SST-2)
        return tokenizer(example['sentence'], padding="max_length", truncation=True)

# Fine-tune BERT base on each dataset
for dataset_name in datasets_to_finetune:
    print(f"Fine-tuning BERT base on {dataset_name.upper()} dataset...")

    # Load prepared dataset from disk
    dataset = load_from_disk(os.path.join(prepared_data_path, dataset_name))

    # Determine the number of labels
    num_labels = 2   # 1 for regression (STS-B), 2 for classification (SST-2, RTE)

    # Load BERT base model from pre-trained weights
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    # Tokenize the dataset
    tokenized_datasets = dataset.map(lambda x: tokenize_function(x, dataset_name), batched=True)

    # Determine the metric for best model based on the dataset

    metric_for_best_model = "eval_loss"  # Classification tasks

    # Define training arguments for fine-tuning BERT base
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
        metric_for_best_model=metric_for_best_model  # Use the appropriate metric
    )

    # Initialize Trainer for fine-tuning
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        tokenizer=tokenizer,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, dataset_name)  # Pass dataset_name to compute_metrics
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

