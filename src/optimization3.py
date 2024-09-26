import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import os
import json
import random
import scipy.stats
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from datasets import load_from_disk
from transformers import BertTokenizer, BertForSequenceClassification

# Import the merge_weights function from the merging script
from merging import merge_weights

# Define paths based on the repo structure
models_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
tinybert_path = os.path.join(models_path, "tinybert")
bert_base_path = os.path.join(models_path, "bert-base")
save_merged_model_path = os.path.join(models_path, "merged_model")

# Ensure directories exist
os.makedirs(save_merged_model_path, exist_ok=True)

# Load dataset and prepare validation dataloader
def get_validation_dataloader(dataset_name):
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "prepared")
    dataset = load_from_disk(os.path.join(data_path, dataset_name, "validation"))
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    def tokenize_function(example):
        return tokenizer(example['sentence1'], example['sentence2'], padding='max_length', truncation=True, max_length=128)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    dataloader = DataLoader(tokenized_dataset, batch_size=8, shuffle=False)
    return dataloader

# Compute metrics (accuracy, f1, Pearson, etc.)
def compute_metrics(preds, labels, dataset_name):
    if dataset_name == 'stsb':
        pearson_corr = scipy.stats.pearsonr(preds[:, 0], labels)[0]
        mse = mean_squared_error(labels, preds[:, 0])
        return {"pearson": pearson_corr, "mse": mse}
    else:
        acc = accuracy_score(labels, preds.argmax(-1))
        f1 = f1_score(labels, preds.argmax(-1), average='weighted')
        return {"accuracy": acc, "f1": f1}

# Function to evaluate the model
def evaluate_model(merged_model, dataloader, dataset_name):
    merged_model.eval()
    preds, true_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = merged_model(input_ids=input_ids, attention_mask=attention_mask)
            preds.append(outputs.logits.cpu().numpy())
            true_labels.append(labels.cpu().numpy())

    preds = torch.tensor(preds).view(-1, merged_model.num_labels).numpy()
    true_labels = torch.tensor(true_labels).view(-1).numpy()

    return compute_metrics(preds, true_labels, dataset_name)

# Function to perform random search for alpha
def random_search_alpha(dataset_name, num_samples, save_merged_model_path, tinybert_path, bert_base_path):
    best_alpha = None
    best_metric = -float('inf')

    # Load TinyBERT and BERT base models from the same directories as in merging.py
    tinybert_model = torch.load(os.path.join(tinybert_path, 'pytorch_model.bin')).to(device)
    bert_base_model = BertForSequenceClassification.from_pretrained(save_merged_model_path)

    # Load validation dataloader
    dataloader = get_validation_dataloader(dataset_name)
    
    for i in range(num_samples):
        alpha = random.uniform(0.0, 1.0)  # Randomly sample alpha between 0 and 1
        
        print(f"Sample {i+1}/{num_samples}: Testing alpha = {alpha}")
        
        # Merge models with current alpha (using imported merge_weights function)
        merged_model = merge_weights(tinybert_model, bert_base_model, alpha).to(device)

        # Evaluate model performance
        metrics = evaluate_model(merged_model, dataloader, dataset_name)
        current_metric = metrics['pearson'] if dataset_name == 'stsb' else metrics['accuracy']
        
        print(f"Alpha {alpha}: metric = {current_metric}")
        
        # Update best alpha and save model if it improves
        if current_metric > best_metric:
            best_alpha = alpha
            best_metric = current_metric
            save_directory = os.path.join(save_merged_model_path, dataset_name)
            os.makedirs(save_directory, exist_ok=True)

            # Save the best model weights
            merged_model.save_pretrained(save_directory)

            # Save the tokenizer as well
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            tokenizer.save_pretrained(save_directory)

            # Save the best alpha and hyperparameters
            with open(os.path.join(save_directory, 'hyperparameters.json'), 'w') as f:
                json.dump({
                    'best_alpha': best_alpha,
                    'best_metric': best_metric,
                    'num_samples': num_samples
                }, f)

            print(f"Saved best model for {dataset_name} with alpha={best_alpha}")

    print(f"Best alpha for {dataset_name}: {best_alpha} with metric: {best_metric}")

# Main function to run random search on multiple datasets
if __name__ == "__main__":
    # Use single device (CPU or CUDA)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define datasets and model paths
    datasets_to_finetune = ['stsb', 'sst2', 'rte']

    # Number of random samples for alpha
    num_samples = 20  # You can increase/decrease this based on how many samples you want to try

    # Perform random search for each dataset
    for dataset in datasets_to_finetune:
        random_search_alpha(dataset, num_samples, save_merged_model_path, tinybert_path, bert_base_path)
