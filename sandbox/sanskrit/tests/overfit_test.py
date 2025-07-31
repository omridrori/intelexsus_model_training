import os
import sys
from transformers import BertConfig, TrainingArguments
import torch
import numpy as np

# Add the model directory to the Python path
script_dir = os.path.dirname(__file__)
sanskrit_dir = os.path.abspath(os.path.join(script_dir, '..'))
model_dir = os.path.join(sanskrit_dir, 'model')
sys.path.append(model_dir)

from training_utils import run_training

# --- Overfitting Test Configuration ---

# 1. Model Configuration
model_config = BertConfig(
    vocab_size=32001,
    hidden_size=256,
    num_hidden_layers=4,
    num_attention_heads=4,
    intermediate_size=1024,
    max_position_embeddings=512,
    type_vocab_size=2,
)

# 2. Training Arguments
training_args = TrainingArguments(
    output_dir="./overfit_test_model",
    overwrite_output_dir=True,
    num_train_epochs=150,
    per_device_train_batch_size=16,  # Increased batch size for better GPU utilization
    gradient_accumulation_steps=1,
    eval_strategy="steps", # Evaluate during training
    eval_steps=20,               # Evaluation frequency
    save_steps=50,
    save_total_limit=2,
    fp16=True,
    logging_steps=10,
    report_to="wandb",
)

def prepare_data_splits(base_data_path, train_split_path, eval_split_path, split_ratio=0.9):
    """Splits the base data file into training and evaluation sets."""
    with open(base_data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    np.random.shuffle(lines)
    
    split_index = int(len(lines) * split_ratio)
    train_lines = lines[:split_index]
    eval_lines = lines[split_index:]
    
    with open(train_split_path, 'w', encoding='utf-8') as f:
        f.writelines(train_lines)
        
    with open(eval_split_path, 'w', encoding='utf-8') as f:
        f.writelines(eval_lines)
    
    print(f"Data split: {len(train_lines)} training lines, {len(eval_lines)} evaluation lines.")

def main():
    """Main function to run the overfitting test."""
    # Paths
    base_data_path = os.path.join(script_dir, 'small_test_data.txt')
    train_split_path = os.path.join(script_dir, 'train_split.txt')
    eval_split_path = os.path.join(script_dir, 'eval_split.txt')
    tokenizer_path = os.path.join(sanskrit_dir, 'tokenizer', 'sanskrit-bert-tokenizer')
    project_name = "sanskrit-overfitting-test-with-eval"

    # Prepare data
    prepare_data_splits(base_data_path, train_split_path, eval_split_path)

    print("--- Starting Overfitting Test ---")
    run_training(
        training_args=training_args,
        model_config=model_config,
        train_dataset_path=train_split_path,
        tokenizer_path=tokenizer_path,
        project_name=project_name,
        eval_dataset_path=eval_split_path
    )
    print("--- Overfitting Test Finished ---")

if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"Found {torch.cuda.device_count()} GPU(s).")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU found. Training will run on CPU.")
    
    main() 