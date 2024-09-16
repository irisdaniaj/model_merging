import torch
from transformers import BertForSequenceClassification, TinyBertForSequenceClassification
import os
import json

# Define paths
models_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
tinybert_path = os.path.join(models_path, "tinybert")
bert_base_path = os.path.join(models_path, "bert-base")
save_merged_model_path = os.path.join(models_path, "merged_model")

# Ensure directories exist
os.makedirs(save_merged_model_path, exist_ok=True)

datasets_to_finetune = ['stsb', 'sst2', 'rte']

# Function to merge models
def merge_weights(tinybert, bert_base, alpha):
    merged_model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    with torch.no_grad():
        for (name_bert_base, param_bert_base), (name_tinybert, param_tinybert) in zip(bert_base.named_parameters(), tinybert.named_parameters()):
            if name_bert_base == name_tinybert:
                merged_param = alpha * param_tinybert + (1 - alpha) * param_bert_base
                merged_model.state_dict()[name_bert_base].copy_(merged_param)
    return merged_model

# Merge models for all datasets and save
def merge_models_for_datasets(datasets, alpha=0.5):
    for dataset_name in datasets:
        tinybert_dataset_path = os.path.join(tinybert_path, f"{dataset_name}_finetuned")
        tinybert_model = TinyBertForSequenceClassification.from_pretrained(tinybert_dataset_path)
        bert_base_model = BertForSequenceClassification.from_pretrained(bert_base_path)
        
        # Merge models with alpha
        merged_model = merge_weights(tinybert_model, bert_base_model, alpha)
        
        # Save the merged model parameters
        save_directory = os.path.join(save_merged_model_path, dataset_name)
        os.makedirs(save_directory, exist_ok=True)
        merged_model.save_pretrained(save_directory)

        # Save the hyperparameters (including alpha)
        hyperparameters = {
            "alpha": alpha
        }
        with open(os.path.join(save_directory, "hyperparameters.json"), 'w') as f:
            json.dump(hyperparameters, f)

        print(f"Saved merged model and hyperparameters for {dataset_name} to {save_directory}")

# Run the merging process
if __name__ == "__main__":
    merge_models_for_datasets(datasets_to_finetune)
