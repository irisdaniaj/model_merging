import torch
from transformers import AutoModelForSequenceClassification, BertTokenizer, BertForSequenceClassification
import os
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
import json
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

# Use only SST-2 and RTE
datasets_to_finetune = ['sst2', 'rte']

# Function to get validation dataloader
def get_validation_dataloader(dataset_name):
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "prepared")
    validation_data_path = os.path.join(data_path, dataset_name, "validation")
    
    # Load the validation dataset from disk
    dataset = load_from_disk(validation_data_path)
    
    # Load tokenizer (this should match the tokenizer used for TinyBERT)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Tokenize the dataset based on its column structure
    if dataset_name == 'sst2':  # SST-2 has a single sentence
        def tokenize_function(example):
            return tokenizer(example['sentence'], padding='max_length', truncation=True, max_length=128)
    elif dataset_name == 'rte':  # RTE has sentence1, sentence2
        def tokenize_function(example):
            return tokenizer(example['sentence1'], example['sentence2'], padding='max_length', truncation=True, max_length=128)
    else:
        raise ValueError(f"Dataset {dataset_name} is not recognized or supported.")

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Ensure labels are present in the dataset
    if 'label' in dataset.column_names:  # Check for 'label' column in the dataset
        tokenized_dataset = tokenized_dataset.rename_column('label', 'labels')  # Rename to match PyTorch expectations
    
    # Set the format to PyTorch for direct use with DataLoader
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    # Create a DataLoader
    dataloader = DataLoader(tokenized_dataset, batch_size=16, shuffle=False)
    
    return dataloader

# Function to evaluate model performance
def evaluate_model(model, dataloader, task='classification'):
    model.eval()
    total, correct = 0, 0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch['input_ids'], batch['labels']
            outputs = model(inputs).logits

            if task == 'classification':
                predictions = torch.argmax(outputs, dim=1)
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return accuracy, f1

# Random search for alpha optimization and saving models
def random_search_alpha(dataloader, dataset_name, num_trials=10):
    best_alpha = None
    best_score = 0  # For classification, higher score (accuracy or f1) is better

    # Initialize models for classification (e.g., 2 labels for binary classification)
    tinybert_model = AutoModelForSequenceClassification.from_pretrained('huawei-noah/TinyBERT_General_6L_768D', num_labels=2)
    bert_base_model = BertForSequenceClassification.from_pretrained(bert_base_path, num_labels=2)
    
    for trial in range(num_trials):
        alpha = random.uniform(0, 1)  # Randomly sample alpha from [0, 1]
        print(f"Trial {trial + 1}/{num_trials}, Testing Alpha: {alpha:.4f}")
        
        # Merge models using the sampled alpha
        merged_model = merge_weights(tinybert_model, bert_base_model, alpha)
        
        # Evaluate the merged model
        accuracy, f1 = evaluate_model(merged_model, dataloader, task='classification')
        score = accuracy  # Use accuracy to track the best alpha
        
        # Save the merged model, hyperparameters, and alpha values after each trial
        trial_save_dir = os.path.join(save_merged_model_path, dataset_name, f"trial_{trial + 1}")
        os.makedirs(trial_save_dir, exist_ok=True)
        
        # Save model weights
        merged_model.save_pretrained(trial_save_dir)
        
        # Save tokenizer (to ensure compatibility if needed later)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        tokenizer.save_pretrained(trial_save_dir)
        
        # Save hyperparameters and alpha value used
        hyperparameters = {
            "alpha": alpha,
            "score": score
        }
        with open(os.path.join(trial_save_dir, "hyperparameters.json"), 'w') as f:
            json.dump(hyperparameters, f)
        
        # Update the best alpha if the current one performs better
        if score > best_score:
            best_score = score
            best_alpha = alpha

    return best_alpha, best_score

# Optimize alpha for each dataset and save models
def optimize_alpha_for_datasets(datasets, num_trials=10):
    for dataset_name in datasets:
        print(f"Optimizing alpha for {dataset_name}...")

        # Load the validation DataLoader for the current dataset
        dataloader = get_validation_dataloader(dataset_name)
        
        # Perform random search for the optimal alpha
        best_alpha, best_score = random_search_alpha(dataloader, dataset_name, num_trials=num_trials)
        
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
