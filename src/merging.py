import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Define model paths
tinybert_finetuned_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "tinybert", "stsb_finetuned")  # Update as necessary
bert_base_model_name = "bert-base-uncased"
merged_model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "merged_model")

# Ensure the merged model directory exists
os.makedirs(merged_model_path, exist_ok=True)

# Load the fine-tuned TinyBERT model
tinybert_model = AutoModelForSequenceClassification.from_pretrained(tinybert_finetuned_path)

# Load the pre-trained BERT base model
bert_base_model = AutoModelForSequenceClassification.from_pretrained(bert_base_model_name, num_labels=tinybert_model.config.num_labels)

# Initialize a new model with the same configuration
merged_model = AutoModelForSequenceClassification.from_pretrained(bert_base_model_name, num_labels=tinybert_model.config.num_labels)

# Function to merge models by averaging their weights
def merge_models(merged_model, model1, model2, alpha=0.5):
    """
    Merge two models by averaging their weights.

    Args:
    merged_model (nn.Module): The model to save the merged weights.
    model1 (nn.Module): The first model (e.g., TinyBERT fine-tuned).
    model2 (nn.Module): The second model (e.g., BERT base pre-trained).
    alpha (float): The weighting factor for merging, 0 <= alpha <= 1.
                   If alpha=0.5, it means a simple average.
    
    Returns:
    nn.Module: The merged model.
    """
    for param_name, param in merged_model.named_parameters():
        if param_name in model1.state_dict() and param_name in model2.state_dict():
            param.data = (alpha * model1.state_dict()[param_name].data +
                          (1 - alpha) * model2.state_dict()[param_name].data)
        else:
            print(f"Parameter {param_name} not found in both models.")
    
    return merged_model

# Merge the models
alpha = 0.5  # Weighting factor for merging
merged_model = merge_models(merged_model, tinybert_model, bert_base_model, alpha)

# Save the merged model
merged_model.save_pretrained(merged_model_path)
print(f"Merged model saved to {merged_model_path}")

# Save the tokenizer (assuming the tokenizer for BERT base is used)
tokenizer = AutoTokenizer.from_pretrained(bert_base_model_name)
tokenizer.save_pretrained(merged_model_path)
print(f"Tokenizer saved to {merged_model_path}")

# Save the merging configuration and hyperparameters
merging_info = {
    "alpha": alpha,
    "tinybert_finetuned_path": tinybert_finetuned_path,
    "bert_base_model_name": bert_base_model_name,
    "merged_model_path": merged_model_path
}

merging_info_path = os.path.join(merged_model_path, "merging_info.json")
with open(merging_info_path, "w") as f:
    json.dump(merging_info, f)

print(f"Merging configuration and hyperparameters saved to {merging_info_path}")

