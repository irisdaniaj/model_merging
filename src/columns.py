from datasets import load_from_disk
import os

# Define the path to the prepared data folder
data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "prepared")

# List of datasets
datasets_to_check = ['stsb', 'sst2', 'rte']

# Loop over each dataset
for dataset_name in datasets_to_check:
    # Construct the path to the validation part of the dataset
    validation_data_path = os.path.join(data_path, dataset_name, "validation")
    
    # Load the validation dataset from disk
    dataset = load_from_disk(validation_data_path)
    
    # Print the dataset name and its column names
    print(f"Dataset: {dataset_name}")
    print(f"Columns: {dataset.column_names}")
    print("-" * 40)
