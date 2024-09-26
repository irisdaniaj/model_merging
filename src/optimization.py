import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import Adam
from datasets import load_from_disk
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
import scipy.stats
import os
import torch.nn as nn 
import numpy as np
import json
from tqdm import tqdm

# Load validation data using Hugging Face datasets
def get_validation_dataloader(dataset_name):
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "prepared")
    validation_data_path = os.path.join(data_path, dataset_name, "validation")
    
    # Load the validation dataset from disk
    dataset = load_from_disk(validation_data_path)
    
    # Load tokenizer (this should match the tokenizer used for TinyBERT)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Tokenize the dataset based on its column structure
    if dataset_name == 'stsb':  # STSB has sentence1, sentence2
        def tokenize_function(example):
            return tokenizer(example['sentence1'], example['sentence2'], padding='max_length', truncation=True, max_length=128)
    elif dataset_name == 'sst2':  # SST-2 has a single sentence
        def tokenize_function(example):
            return tokenizer(example['sentence'], padding='max_length', truncation=True, max_length=128)
    elif dataset_name == 'rte':  # RTE has sentence1, sentence2
        def tokenize_function(example):
            return tokenizer(example['sentence1'], example['sentence2'], padding='max_length', truncation=True, max_length=128)
    else:
        raise ValueError(f"Dataset {dataset_name} is not recognized or supported.")

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Set the format to PyTorch for direct use with DataLoader
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    
    # Create a DataLoader
    dataloader = DataLoader(tokenized_dataset, batch_size=16, shuffle=False)
    
    return dataloader

# Define function to compute metrics
def compute_metrics(eval_pred, dataset_name):
    predictions, labels = eval_pred
    if dataset_name == 'stsb':  # Regression task
        predictions = predictions[:, 0]  # Regression output is a single float per example
        pearson_corr = scipy.stats.pearsonr(predictions, labels)[0]
        mse = mean_squared_error(labels, predictions)
        return {"pearson": pearson_corr, "mse": mse}
    else:  # Classification tasks
        preds = predictions.argmax(-1)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='weighted')
        return {"accuracy": acc, "f1": f1}

# Function to evaluate the model on the validation set
def evaluate_model(model, dataloader, dataset_name):
    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['label']
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            if dataset_name == 'stsb':  # Handle regression task
                # STSB outputs a single regression value per example
                batch_predictions = outputs.logits.squeeze().cpu().numpy()  # Flatten the logits to (batch_size,)
                predictions.extend(batch_predictions)  # Extend the predictions list with this batch
            else:
                # Classification tasks (SST-2, RTE)
                batch_predictions = outputs.logits.cpu().numpy()  # Shape: (batch_size, num_labels)
                predictions.extend(batch_predictions)  # Extend the predictions list with this batch
            
            true_labels.extend(labels.cpu().numpy())  # Collect the true labels from the batch

    # Convert predictions and true_labels to arrays for metric computation
    if dataset_name == 'stsb':
        predictions = np.array(predictions)  # Shape: (num_examples,) for regression
    else:
        predictions = np.array(predictions)  # Shape: (num_examples, num_labels) for classification

    true_labels = np.array(true_labels)  # Shape: (num_examples,)

    return compute_metrics((predictions, true_labels), dataset_name)


# Optimize alpha for the merged models
def optimize_alpha(dataset_name, save_merged_model_path, learning_rate=0.01, epochs=10):
    # Load the pre-merged model
    merged_model_path = os.path.join(save_merged_model_path, dataset_name)
    merged_model = BertForSequenceClassification.from_pretrained(merged_model_path)
    tokenizer = BertTokenizer.from_pretrained(merged_model_path)
    
    # Initialize alpha as a learnable parameter
    alpha = torch.tensor(0.5, requires_grad=True)
    optimizer = Adam([alpha], lr=learning_rate)
    
    # Load validation dataloader
    dataloader = get_validation_dataloader(dataset_name)

    # Define loss function based on the dataset type
    if dataset_name == 'stsb':
        loss_fn = nn.MSELoss()  # Use MSE for regression task (STSB)
    else:
        loss_fn = nn.CrossEntropyLoss()  # Use Cross-Entropy for classification tasks (SST-2, RTE)
    
    best_metric = None
    best_alpha = None
    
    print(f"\nStarting training for {dataset_name}...")

    # Progress bar
    for epoch in tqdm(range(epochs), desc=f"Training {dataset_name}"):
        alpha.data.clamp_(0, 1)  # Ensure alpha is within the range [0, 1]
        
        # Evaluate the model
        metrics = evaluate_model(merged_model, dataloader, dataset_name)
        
        # Print detailed info for each epoch
        if dataset_name in ['sst2', 'rte']:
            print(f"Epoch {epoch + 1}/{epochs} - Alpha: {alpha.item():.4f}, Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
        elif dataset_name == 'stsb':
            print(f"Epoch {epoch + 1}/{epochs} - Alpha: {alpha.item():.4f}, Pearson: {metrics['pearson']:.4f}, MSE: {metrics['mse']:.4f}")

        # Calculate loss for backpropagation
        true_labels = []
        predictions = []

        # Collect all predictions and labels for loss calculation
        merged_model.eval()
        with torch.no_grad():
            for batch in dataloader:
                input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['label']
                outputs = merged_model(input_ids=input_ids, attention_mask=attention_mask)

                if dataset_name == 'stsb':
                    # For regression, outputs.logits should be a single float per example
                    batch_predictions = outputs.logits[:, 0].squeeze()
                else:
                    # For classification, use Cross-Entropy Loss
                    batch_predictions = outputs.logits

                predictions.append(batch_predictions)
                true_labels.append(labels)

        predictions = torch.cat(predictions)
        true_labels = torch.cat(true_labels)

        # Compute loss
        loss = loss_fn(predictions, true_labels)

        merged_model.train()

        # Backpropagation step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Save best model based on the metric
        current_metric = metrics['accuracy'] if dataset_name in ['sst2', 'rte'] else metrics['pearson']
        if best_metric is None or current_metric > best_metric:
            best_metric = current_metric
            best_alpha = alpha.item()

            # Ensure the directory for saving exists
            os.makedirs(os.path.join(merged_model_path, "best_model"), exist_ok=True)
            
            # Save the model parameters
            merged_model.save_pretrained(os.path.join(merged_model_path, "best_model"))
            
            # Save the tokenizer
            tokenizer.save_pretrained(os.path.join(merged_model_path, "best_model"))

        # Backpropagation step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Save the final model and best alpha value
    print(f"Best alpha for {dataset_name}: {best_alpha} with metric: {best_metric}")
    
    # Save best alpha and other hyperparameters
    hyperparameters = {"best_alpha": best_alpha, "best_metric": best_metric}
    with open(os.path.join(merged_model_path, "best_model", "hyperparameters.json"), 'w') as f:
        json.dump(hyperparameters, f)

    print(f"Training for {dataset_name} completed. Best alpha: {best_alpha}")

# Run the optimization process for each dataset
if __name__ == "__main__":
    datasets_to_finetune = ['stsb', 'sst2', 'rte']
    save_merged_model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "merged_model")

    for dataset_name in datasets_to_finetune:
        optimize_alpha(dataset_name, save_merged_model_path)

    print("\nAll trainings completed successfully!")
