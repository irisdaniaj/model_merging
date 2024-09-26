import torch
from transformers import AutoModelForSequenceClassification, BertTokenizer, BertForSequenceClassification
import os
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
import json
import torch.nn as nn
import random
from merging import merge_weights
from datasets import load_from_disk

# Define paths
models_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
tinybert_path = os.path.join(models_path, "tinybert")
bert_base_path = os.path.join(models_path, "bert-base")
save_merged_model_path = os.path.join(models_path, "merged_model")

# Ensure directories exist
os.makedirs(save_merged_model_path, exist_ok=True)

datasets_to_finetune = ['stsb', 'sst2', 'rte']

# Function to get validation dataloader
def get_validation_dataloader(dataset_name):
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "prepared")
    validation_data_path = os.path.join(data_path, dataset_name, "validation")
    
    # Load the validation dataset from disk
    dataset = load_from_disk(validation_data_path)
    
    # Load tokenizer (this should match the tokenizer used for TinyBERT)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Tokenize the dataset based on its column structure
    if dataset_name == 'stsb':  # STSB has sentence1, sentence2
    # Initialize model for regression (1 output)
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

# Function to evaluate model performance
def evaluate_model(model, dataloader, task):
    model.eval()
    total, correct = 0, 0
    all_labels = []
    all_preds = []
 
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch['input_ids'], batch['label']
            outputs = model(inputs).logits

            if task == 'classification':
                predictions = torch.argmax(outputs, dim=1)
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            elif task == 'regression':
                # Ensure the output is correctly shaped (should be a single value per sample)
                predictions = outputs.squeeze()  # Squeeze to ensure correct shape
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
 
    if task == 'classification':
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        return accuracy, f1
    elif task == 'regression':
        # Ensure both y_true and y_pred have the same shape
        if len(all_labels) != len(all_preds):
            raise ValueError(f"Mismatch in number of predictions and labels: {len(all_preds)} != {len(all_labels)}")
        mse = mean_squared_error(all_labels, all_preds) #errore qua
        return mse


# Random search for alpha optimization and saving models
def random_search_alpha(dataloader, task, dataset_name, num_trials=10):
    best_alpha = None
    best_score = float('inf') if task == 'regression' else 0

    # Initialize models based on the dataset
    if dataset_name == 'stsb':
        # Initialize model for regression (1 output)
        tinybert_model = AutoModelForSequenceClassification.from_pretrained('huawei-noah/TinyBERT_General_6L_768D')
        bert_base_model = BertForSequenceClassification.from_pretrained(bert_base_path)

        # Modify the final output layer to ensure it outputs a single value for regression
        tinybert_model.classifier = nn.Linear(tinybert_model.config.hidden_size, 1)
        bert_base_model.classifier = nn.Linear(bert_base_model.config.hidden_size, 1)

    else:
        # Initialize models for classification (e.g., 2 labels for binary classification)
        tinybert_model = AutoModelForSequenceClassification.from_pretrained('huawei-noah/TinyBERT_General_6L_768D', num_labels=2)
        bert_base_model = BertForSequenceClassification.from_pretrained(bert_base_path, num_labels=2)
    
    for trial in range(num_trials):
        alpha = random.uniform(0, 1)  # Randomly sample alpha from [0, 1]
        print(f"Trial {trial + 1}/{num_trials}, Testing Alpha: {alpha:.4f}")
        
        # Merge models using the sampled alpha
        merged_model = merge_weights(tinybert_model, bert_base_model, alpha)
        
        # Evaluate the merged model
        if task == 'classification':
            accuracy, f1 = evaluate_model(merged_model, dataloader, task)
            score = accuracy  # Use accuracy to track the best alpha, or use F1
        elif task == 'regression':
            mse = evaluate_model(merged_model, dataloader, task)
            score = mse  # Lower MSE is better
        
        # Save the merged model, hyperparameters, and alpha values after each trial
        trial_save_dir = os.path.join(save_merged_model_path, dataset_name, f"trial_{trial + 1}")
        os.makedirs(trial_save_dir, exist_ok=True)
        
        # Save model weights
        merged_model.save_pretrained(trial_save_dir)
        
        # Save tokenizer (to ensure compatibility if needed later)
        tokenizer.save_pretrained(trial_save_dir)
        
        # Save hyperparameters and alpha value used
        hyperparameters = {
            "alpha": alpha,
            "score": score
        }
        with open(os.path.join(trial_save_dir, "hyperparameters.json"), 'w') as f:
            json.dump(hyperparameters, f)
        
        # Update the best alpha if the current one performs better
        if (task == 'classification' and score > best_score) or (task == 'regression' and score < best_score):
            best_score = score
            best_alpha = alpha

    return best_alpha, best_score


# Optimize alpha for each dataset and save models
def optimize_alpha_for_datasets(datasets, num_trials=10):
    for dataset_name in datasets:
        task = 'classification' if dataset_name in ['sst2', 'rte'] else 'regression'
        print(f"Optimizing alpha for {dataset_name}...")

        # Load the validation DataLoader for the current dataset
        dataloader = get_validation_dataloader(dataset_name)
        
        # Perform random search for the optimal alpha
        best_alpha, best_score = random_search_alpha(dataloader, task, dataset_name, num_trials=num_trials)
        
        # Log and save the best alpha and score
        print(f"Best Alpha for {dataset_name}: {best_alpha}, Best Score: {best_score}")
        
        # Save the best alpha for future reference
        best_save_dir = os.path.join(save_merged_model_path, dataset_name, "best_model")
        os.makedirs(best_save_dir, exist_ok=True)
        with open(os.path.join(best_save_dir, "best_alpha.json"), 'w') as f:
            json.dump({"best_alpha": best_alpha, "best_score": best_score}, f)
    
    print(f"Optimization complete for all datasets.")

if __name__ == "__main__":
    # Run the optimization for each dataset with 20 trials per dataset
    optimize_alpha_for_datasets(datasets_to_finetune, num_trials=20)
