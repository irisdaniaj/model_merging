import torch
from transformers import AutoModelForSequenceClassification, BertForSequenceClassification, AutoTokenizer
import os
import json

# Define paths
models_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
tinybert_path = os.path.join(models_path, "tinybert")
bert_base_path = os.path.join(models_path, "bert-base")
save_merged_model_path = os.path.join(models_path, "merged_model")

# Ensure directories exist
os.makedirs(save_merged_model_path, exist_ok=True)

datasets_to_finetune = ['sst2', 'rte']

# Function to load alpha from best_alpha.json
def load_best_alpha(dataset_name):
    best_alpha_path = os.path.join(save_merged_model_path, dataset_name, "best_model", "best_alpha.json")
    if not os.path.exists(best_alpha_path):
        raise FileNotFoundError(f"best_alpha.json not found for dataset {dataset_name}")
    
    with open(best_alpha_path, 'r') as f:
        best_alpha_data = json.load(f)
    
    return best_alpha_data.get("best_alpha", 0.5)  # Default to 0.5 if not found

# Function to merge models
def merge_weights(tinybert, bert_base, alpha):
    merged_model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    with torch.no_grad():
        for (name_bert_base, param_bert_base), (name_tinybert, param_tinybert) in zip(bert_base.named_parameters(), tinybert.named_parameters()):
            if name_bert_base == name_tinybert:
                merged_param = alpha * param_tinybert + (1 - alpha) * param_bert_base
                merged_model.state_dict()[name_bert_base].copy_(merged_param)
    return merged_model

# Merge models for all datasets and save in the same folder as the best_alpha.json file
def merge_models_for_datasets(datasets):
    for dataset_name in datasets:
        tinybert_dataset_path = os.path.join(tinybert_path, f"{dataset_name}_finetuned")
        
        # Load TinyBERT with the same hidden size as BERT base
        tinybert_model = AutoModelForSequenceClassification.from_pretrained('huawei-noah/TinyBERT_General_6L_768D')
        tokenizer = AutoTokenizer.from_pretrained('huawei-noah/TinyBERT_General_6L_768D')  # Load TinyBERT tokenizer
        
        bert_base_model = BertForSequenceClassification.from_pretrained(bert_base_path)
        
        # Load the best alpha for this dataset
        best_alpha = load_best_alpha(dataset_name)
        print(f"Using alpha {best_alpha} for merging models for dataset {dataset_name}")

        # Merge models with the loaded alpha
        merged_model = merge_weights(tinybert_model, bert_base_model, best_alpha)
        
        # Save the merged model in the same directory as best_alpha.json (inside checkpoints)
        save_directory = os.path.join(save_merged_model_path, dataset_name, "best_model", "checkpoints")
        os.makedirs(save_directory, exist_ok=True)
        merged_model.save_pretrained(save_directory)  # Save the merged model

        # Save the tokenizer
        tokenizer.save_pretrained(save_directory)

        # Save the hyperparameters (including alpha)
        hyperparameters = {
            "alpha": best_alpha
        }
        with open(os.path.join(save_directory, "hyperparameters.json"), 'w') as f:
            json.dump(hyperparameters, f)

        print(f"Saved merged model and tokenizer for {dataset_name} to {save_directory}")

# Run the merging process
if __name__ == "__main__":
    merge_models_for_datasets(datasets_to_finetune)
