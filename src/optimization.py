import torch
from transformers import AutoModelForSequenceClassification, BertTokenizer, BertForSequenceClassification, AutoTokenizer, AutoConfig, logging as transformers_logging
import os
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
import json
import random
from datasets import load_from_disk
from safetensors.torch import load_file 
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Set transformers logging to show only errors
transformers_logging.set_verbosity_error()
# Paths to the saved merged models

# Define paths
models_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
tinybert_path = os.path.join(models_path, "tinybert")
bert_base_path = os.path.join(models_path, "bert-base")
save_merged_model_path = os.path.join(models_path, "merged_model")

# Ensure directories exist
os.makedirs(save_merged_model_path, exist_ok=True)

# Use only SST-2 and RTE
datasets_to_finetune = ['sst2', 'rte']

# Load model (supporting both safetensors and pytorch_model.bin)
def load_model_from_checkpoint(checkpoint_dir):
    safetensors_file = os.path.join(checkpoint_dir, "model.safetensors")
    #config_file = os.path.join(checkpoint_dir, "config.json")

    # Load model configuration (using AutoConfig to convert the config file into a proper config object)
    config = AutoConfig.from_pretrained(checkpoint_dir)

    model = AutoModelForSequenceClassification.from_config(config)
    state_dict = load_file(safetensors_file)
    model.load_state_dict(state_dict)

    return model

# Find the latest checkpoint (highest numbered) in a directory
def get_latest_checkpoint_path(checkpoint_dir):
    # List all subdirectories in the checkpoint directory
    subdirs = [d for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))]
    
    # Filter directories that start with 'checkpoint-'
    checkpoint_dirs = [d for d in subdirs if d.startswith('checkpoint-')]
    
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    
    # Sort by the numeric part of the checkpoint (e.g., 'checkpoint-156', 'checkpoint-312')
    checkpoint_dirs.sort(key=lambda x: int(x.split('-')[-1]))
    
    # Return the last checkpoint (with the highest number)
    return os.path.join(checkpoint_dir, checkpoint_dirs[-1])


def merge_weights(tinybert, bert_base, alpha):
    merged_model = BertForSequenceClassification.from_pretrained(bert_base_path)  # Use the BERT base configuration
    with torch.no_grad():
        for (name_bert_base, param_bert_base), (name_tinybert, param_tinybert) in zip(bert_base.named_parameters(), tinybert.named_parameters()):
            if name_bert_base == name_tinybert:
                # Merge weights with alpha interpolation
                merged_param = alpha * param_tinybert + (1 - alpha) * param_bert_base
                merged_model.state_dict()[name_bert_base].copy_(merged_param)
    return merged_model

# Merge models for all datasets and save
def merge_models_for_datasets(datasets, alpha=0.5):
    for dataset_name in datasets:
        tinybert_dataset_path = os.path.join(tinybert_path, f"{dataset_name}_finetuned")
        
        # Find the last checkpoint
        latest_checkpoint_path = get_latest_checkpoint_path(tinybert_dataset_path)
        
        # Load TinyBERT finetuned model from the latest checkpoint (either from safetensors or pytorch_model.bin)
        tinybert_model = load_model_from_checkpoint(latest_checkpoint_path)
        tokenizer = AutoTokenizer.from_pretrained(latest_checkpoint_path)  # Load the tokenizer corresponding to TinyBERT finetuned model
        
        # Load pretrained BERT base model
        bert_base_model = BertForSequenceClassification.from_pretrained(bert_base_path)
        
        # Merge models with alpha
        merged_model = merge_weights(tinybert_model, bert_base_model, alpha)
        
        # Save the merged model parameters
        save_directory = os.path.join(save_merged_model_path, dataset_name)
        os.makedirs(save_directory, exist_ok=True)
        
        # Save the merged model (weights and config)
        merged_model.save_pretrained(save_directory)
        
        # Save the tokenizer (TinyBERT's tokenizer)
        tokenizer.save_pretrained(save_directory)

        # Save the hyperparameters (including alpha)
        hyperparameters = {
            "alpha": alpha,
            "dataset": dataset_name
        }
        with open(os.path.join(save_directory, "hyperparameters.json"), 'w') as f:
            json.dump(hyperparameters, f)

        print(f"Saved merged model, tokenizer, and hyperparameters for {dataset_name} to {save_directory}")

# Function to get validation dataloader
def get_validation_dataloader(dataset_name):
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "prepared")
    validation_data_path = os.path.join(data_path, dataset_name, "validation")
    
    # Load the validation dataset from disk
    dataset = load_from_disk(validation_data_path)
    
    # Load tokenizer (this also match the tokenizer used for TinyBERT)
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
def random_search_alpha(dataloader, dataset_name, checkpoint_dir, num_trials):
    best_alpha = None
    best_score = 0  # For classification, higher score (accuracy or f1) is better
    best_model = None

    # Initialize models for classification (e.g., 2 labels for binary classification)
    bert_base_model = BertForSequenceClassification.from_pretrained(bert_base_path, num_labels=2)
    tinybert_model = load_model_from_checkpoint(checkpoint_dir)

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
            best_model = merged_model

    return best_alpha, best_score, best_model

# Optimize alpha for each dataset and save models
def optimize_alpha_for_datasets(datasets, num_trials):
    for dataset_name in datasets:
        print(f"Optimizing alpha for {dataset_name}...")

        # Load the validation DataLoader for the current dataset
        dataloader = get_validation_dataloader(dataset_name)
        tinybert_dataset_path = os.path.join(tinybert_path, f"{dataset_name}_finetuned")
        
        # Find the last checkpoint for this dataset
        checkpoint_dir = get_latest_checkpoint_path(tinybert_dataset_path)
        # Perform random search for the optimal alpha
        best_alpha, best_score, best_model = random_search_alpha(dataloader, dataset_name, checkpoint_dir, num_trials=num_trials)
        
        # Log and save the best alpha and score
        print(f"Best Alpha for {dataset_name}: {best_alpha}, Best Score: {best_score}")
        
        # Save the best alpha for future reference
        best_save_dir = os.path.join(save_merged_model_path, dataset_name, "best_model")
        os.makedirs(best_save_dir, exist_ok=True)
        with open(os.path.join(best_save_dir, "best_alpha.json"), 'w') as f:
            json.dump({"best_alpha": best_alpha, "best_score": best_score}, f)

        
        # Save the merged model in the same directory as best_alpha.json (inside checkpoints)
        save_directory = os.path.join(save_merged_model_path, dataset_name, "best_model", "checkpoints")
        os.makedirs(save_directory, exist_ok=True)
        best_model.save_pretrained(save_directory)  # Save the merged model
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir) 
        # Save the tokenizer
        tokenizer.save_pretrained(save_directory)

        # Save the hyperparameters (including alpha)
        hyperparameters = {
            "alpha": best_alpha
        }
        with open(os.path.join(save_directory, "hyperparameters.json"), 'w') as f:
            json.dump(hyperparameters, f)

        print(f"Saved merged model and tokenizer for {dataset_name} to {save_directory}")
    
    print(f"Optimization complete for all datasets.")

if __name__ == "__main__":
    # Run the optimization for each dataset with 20 trials per dataset
    optimize_alpha_for_datasets(datasets_to_finetune, num_trials=20)
