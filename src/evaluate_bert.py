import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
from datasets import load_dataset
from tqdm import tqdm
from torchprofile import profile_macs  # To calculate FLOPs (FLOPs = 2 * MACs)
import json

# Define paths to the saved fine-tuned models
base_model_path = "/path/to/your/models"  # Update this to the base path where the models are saved
sst2_checkpoint = os.path.join(base_model_path, "sst2_finetuned", "checkpoint-12630")  # Use your desired checkpoint

# Function to load model and tokenizer
def load_model_and_tokenizer(checkpoint_path):
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    return model, tokenizer

# Function to calculate FLOPs (Floating Point Operations)
def calculate_flops(model, inputs):
    macs = profile_macs(model, inputs)
    flops = 2 * macs  # FLOPs = 2 * MACs (Multiply-Accumulate Operations)
    return flops

# Inference and accuracy calculation on IMDb dataset
def run_inference_imdb():
    print(f"Running inference on IMDb Movie Reviews dataset")

    # Load the fine-tuned SST-2 model and tokenizer
    model, tokenizer = load_model_and_tokenizer(sst2_checkpoint)

    # Load IMDb dataset
    dataset = load_dataset("imdb")
    test_dataset = dataset["test"]

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
    results = {
        "accuracy": accuracy,
        "total_flops": total_flops
    }

    # Define path to save results
    results_save_path = "inference_results.json"
    with open(results_save_path, "w") as f:
        json.dump(results, f)

    print(f"Results saved to {results_save_path}")

# Run inference and calculate accuracy and FLOPs
run_inference_imdb()
