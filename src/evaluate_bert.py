import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
from datasets import load_dataset
from tqdm import tqdm
from torchprofile import profile_macs  # To calculate FLOPs (FLOPs = 2 * MACs)
import json

# Define paths to the saved fine-tuned models
base_model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "bert-base")
sst2_checkpoint = os.path.join(base_model_path, "sst2_finetuned", "checkpoint-12630")  # Use your desired checkpoint
rte_checkpoint = os.path.join(base_model_path, "rte_finetuned", "checkpoint-468")  # Use your desired checkpoint

# Function to load model and tokenizer
def load_model_and_tokenizer(checkpoint_path):
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    return model, tokenizer

# Function to calculate FLOPs (Floating Point Operations)
def calculate_flops(model, inputs):
    # Extract the input_ids tensor (and optionally attention_mask)
    input_tensor = inputs["input_ids"]
    
    # Forward pass through the model to measure FLOPs
    macs = profile_macs(model, (input_tensor,))
    flops = 2 * macs  # FLOPs = 2 * MACs (Multiply-Accumulate Operations)
    return flops

# Inference and accuracy calculation on IMDb dataset
def run_inference_imdb(subsample_size):
    print(f"Running inference on IMDb Movie Reviews dataset")

    # Load the fine-tuned SST-2 model and tokenizer
    model, tokenizer = load_model_and_tokenizer(sst2_checkpoint)

    # Load IMDb dataset
    dataset = load_dataset("imdb")
    test_dataset = dataset["test"].shuffle(seed=42).select(range(subsample_size))  


    # Set model to evaluation mode
    model.eval()

    # Initialize variables for accuracy and FLOPs
    correct_predictions = 0
    total_predictions = 0
    total_flops = 0

    # Run inference and calculate accuracy and FLOPs
    for example in tqdm(test_dataset):
        input_sentence = example["text"]
        true_label = example["label"]

        # Tokenize the input sentence
        inputs = tokenizer(input_sentence, return_tensors="pt", padding="max_length", truncation=True, max_length=128)

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
    print(f"Accuracy on IMDb test set: {accuracy:.2f}%")

    # Report FLOPs
    print(f"Total FLOPs for inference on IMDb test set: {total_flops:.2e} FLOPs")

    # Save results to a JSON file
    results_filename = "imdb_bert_results.json"
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

    print(f"Results saved to {results_save_path}")

def run_inference_snli(subsample_size):
    print(f"Running inference on SNLI dataset using RTE-finetuned BERT")

    # Load the fine-tuned RTE model and tokenizer
    model, tokenizer = load_model_and_tokenizer(rte_checkpoint)

    # Load SNLI dataset
    dataset = load_dataset("snli")
    test_dataset = dataset["test"].shuffle(seed=42).select(range(subsample_size)) 


    # Set model to evaluation mode
    model.eval()

    # Initialize variables for accuracy and FLOPs
    correct_predictions = 0
    total_predictions = 0
    total_flops = 0

    # Run inference and calculate accuracy and FLOPs
    for example in tqdm(test_dataset):
        premise = example["premise"]
        hypothesis = example["hypothesis"]
        true_label = example["label"]  # 0: entailment, 1: neutral, 2: contradiction

        # Some examples in SNLI have label '-1' which means that the label is not available (ignore these)
        if true_label == -1:
            continue

        # Tokenize the input sentence pair
        inputs = tokenizer(premise, hypothesis, return_tensors="pt", padding="max_length", truncation=True, max_length=128)

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
    print(f"Accuracy on SNLI test set: {accuracy:.2f}%")

    # Report FLOPs
    print(f"Total FLOPs for inference on SNLI test set: {total_flops:.2e} FLOPs")

    # Save results to a JSON file
    results_filename = "snli_bert_results.json"
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

    print(f"Results saved to {results_save_path}")
# Main function to execute the scripts
if __name__ == "__main__":
    run_inference_imdb(subsample_size = 5000)
    run_inference_snli(subsample_size = 5000)
