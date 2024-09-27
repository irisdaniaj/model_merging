from datasets import load_dataset
import os

# Define GLUE datasets to download
datasets_to_download = [ 'sst2', 'rte']

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the base directory (two levels up from the script directory)
base_dir = os.path.dirname(script_dir)

# Define the datasets folder path (inside the parent directory)
data_dir = os.path.join(base_dir, "data")

# Ensure the datasets directory exists
os.makedirs(data_dir, exist_ok=True)

# Function to download and save datasets
def download_and_save_dataset(dataset_name):
    print(f"Downloading {dataset_name.upper()} dataset...")
    dataset = load_dataset('glue', dataset_name)
    dataset.save_to_disk(os.path.join(data_dir, dataset_name))
    print(f"{dataset_name.upper()} dataset downloaded and saved locally in {os.path.join(data_dir, dataset_name)}!")

# Download the required datasets
for dataset_name in datasets_to_download:
    download_and_save_dataset(dataset_name)

print("All datasets are downloaded and saved locally!")
