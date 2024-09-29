import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, logging as transformers_logging
import os
from datasets import load_dataset
from tqdm import tqdm
import json
from torchprofile import profile_macs  # To calculate MACs, from which we can derive FLOPs (2 * MACs)
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Set transformers logging to show only errors
transformers_logging.set_verbosity_error()
# Paths to the saved merged models
models_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "merged_model")
snli_model_checkpoint = os.path.join(models_path, "rte", "best_model", "checkpoints") 
mrpc_model_checkpoint = os.path.join(models_path, "rte", "best_model", "checkpoints") # Adjust this if necessary
imdb_model_checkpoint = os.path.join(models_path, "sst2", "best_model", "checkpoints")  # Adjust this if necessary

# Function to load model and tokenizer
def load_model_and_tokenizer(checkpoint_path):
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    return model, tokenizer

# Function to calculate FLOPs (Floating Point Operations)
def calculate_flops(model, inputs):
    # Extract the input_ids tensor (and optionally attention_mask)
    input_tensor = inputs["input_ids"]
    
    # Forward pass through the model to measure MACs
    macs = profile_macs(model, (input_tensor,))
    flops = 2 * macs  # FLOPs = 2 * MACs (Multiply-Accumulate Operations)
    return flops

# Function to run inference on a dataset
def run_inference(dataset, model, tokenizer, task='classification'):
    model.eval()

    # Initialize variables for accuracy and FLOPs calculation
    correct_predictions = 0
    total_predictions = 0
    total_flops = 0

    # Run inference
    for example in tqdm(dataset):
        if task == 'classification':
            input_text = example['text'] if 'text' in example else example['sentence']  # For IMDb, 'text' key is used
            inputs = tokenizer(input_text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
            true_label = example['label']

        elif task == 'nli':  # Natural Language Inference (for SNLI)
            premise = example["premise"]
            hypothesis = example["hypothesis"]
            inputs = tokenizer(premise, hypothesis, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
            true_label = example['label']

            # Ignore examples with label '-1' in SNLI (invalid label)
            if true_label == -1:
                continue

        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()

        # Check if prediction matches true label
        if predicted_class == true_label:
            correct_predictions += 1
        total_predictions += 1

        # Calculate FLOPs for this inference
        flops = calculate_flops(model, inputs)
        total_flops += flops

    # Calculate accuracy
    accuracy = correct_predictions / total_predictions * 100
    return accuracy, total_flops

# Function to run inference on the SNLI dataset
def run_inference_snli():
    print("Running inference on SNLI using the merged model (RTE-finetuned)")

    # Load the SNLI dataset
    snli_dataset = load_dataset("snli", split="test").shuffle(seed=42).select(range(5000))  # Subsample of 10,000 for efficiency

    # Load the merged RTE model for NLI task
    model, tokenizer = load_model_and_tokenizer(snli_model_checkpoint)

    # Run inference on the SNLI dataset
    accuracy, total_flops = run_inference(snli_dataset, model, tokenizer, task='nli')

    print(f"Accuracy on SNLI test set: {accuracy:.2f}%")
    print(f"Total FLOPs for SNLI test set: {total_flops:.2e} FLOPs")

    # Save results
    results_filename = "snli_merged_results.json"
    results_save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", results_filename)

    # Save results to the correct path
    results = {
        "accuracy": accuracy,
        "total_flops": total_flops
    }

    # Ensure the 'results' directory exists, if not create it
    os.makedirs(os.path.dirname(results_save_path), exist_ok=True)

    with open(results_save_path, "w") as f:
        json.dump(results, f)

    print(f"SNLI Results saved to {results_save_path}")

# Function to run inference on the IMDb dataset
def run_inference_imdb():
    print("Running inference on IMDb using the merged model (SST-2-finetuned)")

    # Load the IMDb dataset
    imdb_dataset = load_dataset("imdb", split="test").shuffle(seed=42).select(range(5000))  # Subsample of 10,000 for efficiency

    # Load the merged SST-2 model for sentiment analysis
    model, tokenizer = load_model_and_tokenizer(imdb_model_checkpoint)

    # Run inference on the IMDb dataset
    accuracy, total_flops = run_inference(imdb_dataset, model, tokenizer, task='classification')

    print(f"Accuracy on IMDb test set: {accuracy:.2f}%")
    print(f"Total FLOPs for IMDb test set: {total_flops:.2e} FLOPs")

    # Save results
    results_filename = "imdb_merged_results.json"
    results_save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", results_filename)

    # Save results to the correct path
    results = {
        "accuracy": accuracy,
        "total_flops": total_flops
    }

    # Ensure the 'results' directory exists, if not create it
    os.makedirs(os.path.dirname(results_save_path), exist_ok=True)

    with open(results_save_path, "w") as f:
        json.dump(results, f)

    print(f"IMDb Results saved to {results_save_path}")

def run_inference_mrpc():
    print("Running inference on MRPC dataset using the merged model")

    # Load the MRPC dataset from the GLUE benchmark
    mrpc_dataset = load_dataset("glue", "mrpc", split="test").shuffle(seed=42) #.select(range(500))  # Use a subsample for efficiency

    # Load the merged MRPC model for paraphrase classification
    model, tokenizer = load_model_and_tokenizer(mrpc_model_checkpoint)

    # Set model to evaluation mode
    model.eval()

    # Initialize variables for accuracy and FLOPs calculation
    correct_predictions = 0
    total_predictions = 0
    total_flops = 0

    # Run inference on the MRPC dataset
    for example in tqdm(mrpc_dataset):
        sentence1 = example["sentence1"]
        sentence2 = example["sentence2"]
        true_label = example["label"]  # 1: paraphrase, 0: not paraphrase

        # Tokenize the input sentence pair
        inputs = tokenizer(sentence1, sentence2, return_tensors="pt", padding="max_length", truncation=True, max_length=128)

        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()

            # Calculate accuracy
            if predicted_class == true_label:
                correct_predictions += 1
            total_predictions += 1

            # Calculate FLOPs for this inference
            flops = calculate_flops(model, inputs)
            total_flops += flops

    # Calculate final accuracy
    accuracy = correct_predictions / total_predictions * 100
    print(f"Accuracy on MRPC test set: {accuracy:.2f}%")

    # Report FLOPs
    print(f"Total FLOPs for MRPC test set: {total_flops:.2e} FLOPs")

    # Save results to a JSON file
    results_filename = "mrpc_merged_results.json"
    results_save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", results_filename)

    # Save results to the correct path
    results = {
        "accuracy": accuracy,
        "total_flops": total_flops
    }

    # Ensure the 'results' directory exists, if not create it
    os.makedirs(os.path.dirname(results_save_path), exist_ok=True)

    with open(results_save_path, "w") as f:
        json.dump(results, f)

    print(f"MRPC Results saved to {results_save_path}")

if __name__ == "__main__":
    # Run inference on SNLI
    #run_inference_snli()
    #run_inference_imdb()
    run_inference_mrpc()