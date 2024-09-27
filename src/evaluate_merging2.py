import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, logging as transformers_logging
import os
from datasets import load_from_disk
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
rte_model_checkpoint = os.path.join(models_path, "rte", "best_model", "checkpoints")  # Adjust this if necessary
sst2_model_checkpoint = os.path.join(models_path, "sst2", "best_model", "checkpoints")  # Adjust this if necessary

# Define data paths based on the folder structure
sst2_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "prepared", "sst2")
rte_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "prepared", "rte")

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
            input_text = example['sentence']
            inputs = tokenizer(input_text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
            true_label = example['label']

        elif task == 'nli':  # Natural Language Inference (for RTE)
            sentence1 = example["sentence1"]
            sentence2 = example["sentence2"]
            inputs = tokenizer(sentence1, sentence2, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
            true_label = example['label']

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

# Function to run inference on the RTE dataset
def run_inference_rte():
    print("Running inference on RTE using the merged model")

    # Load the RTE dataset from the prepared folder
    rte_dataset = load_from_disk(rte_data_path)['test']

    # Load the merged model for RTE task
    model, tokenizer = load_model_and_tokenizer(rte_model_checkpoint)

    # Run inference on the RTE dataset
    accuracy, total_flops = run_inference(rte_dataset, model, tokenizer, task='nli')

    print(f"Accuracy on RTE test set: {accuracy:.2f}%")
    print(f"Total FLOPs for RTE test set: {total_flops:.2e} FLOPs")

    # Save results
    results_filename = "rte_inference_merged_results.json"
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

    print(f"RTE Results saved to {results_save_path}")

# Function to run inference on the SST-2 dataset
def run_inference_sst2():
    print("Running inference on SST-2 using the merged model")

    # Load the SST-2 dataset from the prepared folder
    sst2_dataset = load_from_disk(sst2_data_path)['test']

    # Load the merged model for sentiment analysis task (SST-2)
    model, tokenizer = load_model_and_tokenizer(sst2_model_checkpoint)

    # Run inference on the SST-2 dataset
    accuracy, total_flops = run_inference(sst2_dataset, model, tokenizer, task='classification')

    print(f"Accuracy on SST-2 test set: {accuracy:.2f}%")
    print(f"Total FLOPs for SST-2 test set: {total_flops:.2e} FLOPs")

    # Save results
    results_filename = "sst2_inference_merged_results.json"
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

    print(f"SST-2 Results saved to {results_save_path}")

if __name__ == "__main__":
    # Run inference on RTE
    run_inference_rte()

    # Run inference on SST-2
    run_inference_sst2()
