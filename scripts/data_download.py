from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
import os

# Define the model names for BERT base uncased and TinyBERT uncased
bert_base_model_name = 'bert-base-uncased'
tinybert_model_name = 'huawei-noah/TinyBERT_General_4L_312D'

# Define GLUE datasets to download
datasets_to_download = ['sts-b', 'sst2', 'rte']

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the base directory (two levels up from the script directory)
base_dir = os.path.dirname(os.path.dirname(script_dir))

# Define the models and datasets folder paths (inside the parent directory)
models_dir = os.path.join(base_dir, "models")
data_dir = os.path.join(base_dir, "data")

# Ensure the models and datasets directories exist
os.makedirs(models_dir, exist_ok=True)
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

# Download and cache the BERT base uncased model and tokenizer
print(f"Downloading {bert_base_model_name}...")
bert_base_model = AutoModel.from_pretrained(bert_base_model_name)
bert_base_tokenizer = AutoTokenizer.from_pretrained(bert_base_model_name)
bert_base_save_path = os.path.join(models_dir, 'bert-base-uncased')

# Save BERT base uncased model and tokenizer locally
bert_base_model.save_pretrained(bert_base_save_path)
bert_base_tokenizer.save_pretrained(bert_base_save_path)
print(f"{bert_base_model_name} model and tokenizer saved locally at {bert_base_save_path}!")

# Download and cache the TinyBERT model and tokenizer
print(f"Downloading {tinybert_model_name}...")
tinybert_model = AutoModel.from_pretrained(tinybert_model_name)
tinybert_tokenizer = AutoTokenizer.from_pretrained(tinybert_model_name)
tinybert_save_path = os.path.join(models_dir, 'tinybert-uncased')

# Save TinyBERT model and tokenizer locally
tinybert_model.save_pretrained(tinybert_save_path)
tinybert_tokenizer.save_pretrained(tinybert_save_path)
print(f"{tinybert_model_name} model and tokenizer saved locally at {tinybert_save_path}!")

print("All models and datasets are downloaded and saved locally!")
