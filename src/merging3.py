import torch
from transformers import AutoModelForSequenceClassification, BertForSequenceClassification, AutoTokenizer, AutoConfig
import os
import json
from safetensors.torch import load_file  # Import safetensors

# Define paths
models_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
tinybert_path = os.path.join(models_path, "tinybert")
bert_base_path = os.path.join(models_path, "bert-base")
save_merged_model_path = os.path.join(models_path, "merged_model")

# Ensure directories exist
os.makedirs(save_merged_model_path, exist_ok=True)

datasets_to_finetune = ['sst2', 'rte']

# Function to merge models
def merge_weights(tinybert, bert_base, alpha):
    merged_model = BertForSequenceClassification.from_pretrained(bert_base_path)  # Use the BERT base configuration
    with torch.no_grad():
        for (name_bert_base, param_bert_base), (name_tinybert, param_tinybert) in zip(bert_base.named_parameters(), tinybert.named_parameters()):
            if name_bert_base == name_tinybert:
                # Merge weights with alpha interpolation
                merged_param = alpha * param_tinybert + (1 - alpha) * param_bert_base
                merged_model.state_dict()[name_bert_base].copy_(merged_param)
    return merged_model

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

# Run the merging process
if __name__ == "__main__":
    merge_models_for_datasets(datasets_to_finetune)
