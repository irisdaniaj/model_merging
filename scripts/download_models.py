from transformers import AutoModel, AutoTokenizer
import os

# Define the model names for BERT base and TinyBERT
bert_base_model_name = 'bert-base-uncased'
tinybert_model_name = 'huawei-noah/TinyBERT_General_4L_312D'

# Download and cache the BERT base model and tokenizer
print(f"Downloading {bert_base_model_name}...")
bert_base_model = AutoModel.from_pretrained(bert_base_model_name)
bert_base_tokenizer = AutoTokenizer.from_pretrained(bert_base_model_name)
print(f"{bert_base_model_name} downloaded successfully!")

# Download and cache the TinyBERT model and tokenizer
print(f"Downloading {tinybert_model_name}...")
tinybert_model = AutoModel.from_pretrained(tinybert_model_name)
tinybert_tokenizer = AutoTokenizer.from_pretrained(tinybert_model_name)
print(f"{tinybert_model_name} downloaded successfully!")

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the base directory (one level up from the script directory)
base_dir = os.path.dirname(script_dir)

# Define the models folder path (inside the parent directory)
models_dir = os.path.join(base_dir, "models")

# Ensure the models directory exists
os.makedirs(models_dir, exist_ok=True)

# Paths to save the models and tokenizers
bert_base_save_path = os.path.join(models_dir, 'bert-base')
tinybert_save_path = os.path.join(models_dir, 'tinybert')

# Save BERT base model and tokenizer locally
bert_base_model.save_pretrained(bert_base_save_path)
bert_base_tokenizer.save_pretrained(bert_base_save_path)
print(f"{bert_base_model_name} model and tokenizer saved locally at {bert_base_save_path}!")

# Save TinyBERT model and tokenizer locally
tinybert_model.save_pretrained(tinybert_save_path)
tinybert_tokenizer.save_pretrained(tinybert_save_path)
print(f"{tinybert_model_name} model and tokenizer saved locally at {tinybert_save_path}!")

print("Both models and tokenizers are downloaded and saved locally!")
