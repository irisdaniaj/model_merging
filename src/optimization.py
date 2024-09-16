import torch
from transformers import BertForSequenceClassification, BertTokenizer
from torch.optim import Adam
from datasets import load_from_disk
from torch.utils.data import DataLoader
import os
import json
from sklearn.metrics import accuracy_score

# Define paths
models_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
save_merged_model_path = os.path.join(models_path, "merged_model")
data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "prepared")
datasets_to_finetune = ['stsb', 'sst2', 'rte']

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

# Function to evaluate the model on the validation set
def evaluate_model(model, dataloader):
    model.eval()
    predictions, true_labels = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['label']
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Apply argmax to get the predicted class labels (discrete values)
            preds = torch.argmax(outputs.logits, dim=-1)
            
            # Convert predictions and labels to lists for evaluation
            predictions.extend(preds.cpu().tolist())  # Move to CPU and convert to list
            true_labels.extend(labels.cpu().tolist())  # Move to CPU and convert to list
    
    # Now calculate accuracy using sklearn's accuracy_score
    accuracy = accuracy_score(true_labels, predictions)
    return accuracy

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
    
    best_alpha = None
    best_accuracy = 0
    
    for epoch in range(epochs):
        alpha.data.clamp_(0, 1)  # Ensure alpha is within the range [0, 1]
        
        accuracy = evaluate_model(merged_model, dataloader)
        print(f"Dataset: {dataset_name}, Epoch {epoch + 1}, alpha = {alpha.item()}, accuracy = {accuracy}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_alpha = alpha.item()
        
        # Loss is negative accuracy since we want to maximize accuracy
        loss = -accuracy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Save the best alpha
    hyperparameters = {"best_alpha": best_alpha}
    hyperparameters_path = os.path.join(merged_model_path, "hyperparameters.json")
    with open(hyperparameters_path, 'w') as f:
        json.dump(hyperparameters, f)

    print(f"Best alpha for {dataset_name}: {best_alpha} saved at {hyperparameters_path}")

# Run the optimization process for each dataset
if __name__ == "__main__":
    for dataset_name in datasets_to_finetune:
        optimize_alpha(dataset_name, save_merged_model_path)
