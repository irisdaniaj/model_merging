import os
from datasets import load_from_disk

# Define paths
data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
prepared_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "prepared")

# Ensure the prepared data directory exists
os.makedirs(prepared_data_path, exist_ok=True)

# Function for minimal data preparation
def prepare_dataset(dataset_name):
    print(f"Preparing {dataset_name.upper()} dataset...")

    # Load dataset from disk
    dataset = load_from_disk(os.path.join(data_path, dataset_name))

    dataset = dataset.map(lambda example: {'label': int(example['label'])})

    # Save the prepared datasets back to disk
    prepared_dataset_path = os.path.join(prepared_data_path, dataset_name)
    dataset.save_to_disk(prepared_dataset_path)

    print(f"Prepared {dataset_name.upper()} dataset saved to {prepared_dataset_path}.")

# List of datasets to prepare
datasets_to_prepare = [ 'sst2', 'rte']

# Prepare each dataset
for dataset_name in datasets_to_prepare:
    prepare_dataset(dataset_name)

print("Data preparation complete for all datasets!")
