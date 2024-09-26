import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import Adam
from datasets import load_from_disk
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from scipy.stats import spearmanr
import os
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

    tokenized_dataset = dataset.map(tokenize_function, batched=True) #map = applies tokenization to every element in the dataset 
    
    # Set the format to PyTorch for direct use with DataLoader
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label']) #convert columns in pytorhc tensors
    
    # Create a DataLoader
    dataloader = DataLoader(tokenized_dataset, batch_size=16, shuffle=False)
    
    return dataloader

# Function to evaluate the model on the validation set
def evaluate_model(model, dataloader, dataset_name):
    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['label']
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            if dataset_name in ['sst2', 'rte']:  # Classification tasks
                preds = torch.argmax(outputs.logits, dim=-1)
                predictions.extend(preds.cpu().tolist())
                true_labels.extend(labels.cpu().tolist())
            elif dataset_name == 'stsb':  # Regression task (STS-B)
                preds = outputs.logits.squeeze(-1).cpu().tolist()  # Squeeze to make predictions 1D
                predictions.extend(preds)
                true_labels.extend(labels.cpu().tolist())

    # Compute metrics based on the dataset
    if dataset_name in ['sst2', 'rte']:  # For SST-2 and RTE, use accuracy
        accuracy = accuracy_score(true_labels, predictions)
        return accuracy
    elif dataset_name == 'stsb':  # For STS-B, use Spearman correlation
        print("labels", true_labels, "pred", predictions)
        spearman_corr = spearmanr(true_labels, predictions)
        return spearman_corr[0]  # Return only the Spearman correlation coefficient


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
            print(f"Epoch {epoch + 1}/{epochs} - Alpha: {alpha.item():.4f}, Accuracy: {metrics:.4f}")
        elif dataset_name == 'stsb':
            print(f"Epoch {epoch + 1}/{epochs} - Alpha: {alpha.item():.4f}, Spearman: {spearmanr}")

        # Save best model based on the metric
        current_metric = metrics  # For STSB, it's already the correlation coefficient
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
        loss = -current_metric
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