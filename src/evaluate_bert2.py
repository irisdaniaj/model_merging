import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, logging as transformers_logging
import os
from datasets import load_from_disk
from tqdm import tqdm
from torchprofile import profile_macs  # To calculate FLOPs (FLOPs = 2 * MACs)
import json
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Set transformers logging to show only errors
transformers_logging.set_verbosity_error()

# Define paths to the saved fine-tuned models
base_model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "bert-base")
sst2_checkpoint = os.path.join(base_model_path, "sst2_finetuned", "checkpoint-12630")  # Use your desired checkpoint
rte_checkpoint = os.path.join(base_model_path, "rte_finetuned", "checkpoint-468")  # Use your desired checkpoint

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
    input_tensor = inputs["input_ids"]
    macs = profile_macs(model, (input_tensor,))
    flops = 2 * macs  # FLOPs = 2 * MACs (Multiply-Accumulate Operations)
    return flops

# Inference and accuracy calculation on SST-2 dataset
def run_inference_sst2():
    print(f"Running inference on SST-2 dataset")

    # Load the fine-tuned SST-2 model and tokenizer
    model, tokenizer = load_model_and_tokenizer(sst2_checkpoint)

    # Load SST-2 dataset from the prepared folder
    dataset = load_from_disk(sst2_data_path)
    test_dataset = dataset['test']

    model.eval()
    correct_predictions = 0
    total_predictions = 0
    total_flops = 0

    for example in tqdm(test_dataset):
        input_sentence = example["sentence"]
        true_label = example["label"]

        inputs = tokenizer(input_sentence, return_tensors="pt", padding="max_length", truncation=True, max_length=128)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()

            if predicted_class == true_label:
                correct_predictions += 1
            total_predictions += 1

            flops = calculate_flops(model, inputs)
            total_flops += flops

    accuracy = correct_predictions / total_predictions * 100
    print(f"Accuracy on SST-2 test set: {accuracy:.2f}%")
    print(f"Total FLOPs for inference on SST-2 test set: {total_flops:.2e} FLOPs")

    results_filename = "sst2_inference_results.json"
    results_save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", results_filename)
    results = {"accuracy": accuracy, "total_flops": total_flops}
    os.makedirs(os.path.dirname(results_save_path), exist_ok=True)

    with open(results_save_path, "w") as f:
        json.dump(results, f)

    print(f"Results saved to {results_save_path}")


# Inference and accuracy calculation on RTE dataset
def run_inference_rte():
    print(f"Running inference on RTE dataset")

    # Load the fine-tuned SST-2 model and tokenizer
    model, tokenizer = load_model_and_tokenizer(rte_checkpoint)

    # Load SST-2 dataset from the prepared folder
    dataset = load_from_disk(rte_data_path)
    test_dataset = dataset['test']

    model.eval()
    correct_predictions = 0
    total_predictions = 0
    total_flops = 0

    for example in tqdm(test_dataset):
        sentence1 = example["sentence1"]
        sentence2 = example["sentence2"]
        true_label = example["label"]

        inputs = tokenizer(sentence1, sentence2, return_tensors="pt", padding="max_length", truncation=True, max_length=128)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()

            if predicted_class == true_label:
                correct_predictions += 1
            total_predictions += 1

            flops = calculate_flops(model, inputs)
            total_flops += flops

    accuracy = correct_predictions / total_predictions * 100
    print(f"Accuracy on RTE test set: {accuracy:.2f}%")
    print(f"Total FLOPs for inference on RTE test set: {total_flops:.2e} FLOPs")

    results_filename = "rte_inference_results.json"
    results_save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", results_filename)
    results = {"accuracy": accuracy, "total_flops": total_flops}
    os.makedirs(os.path.dirname(results_save_path), exist_ok=True)

    with open(results_save_path, "w") as f:
        json.dump(results, f)

    print(f"Results saved to {results_save_path}")

if __name__ == "__main__":
    run_inference_sst2()
    run_inference_rte()
