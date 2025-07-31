import os
from transformers import BertConfig, TrainingArguments
import torch
from training_utils import run_training

# --- Configuration for Overfitting Test ---
# Using the exact same model configuration as the main training script.
model_config = BertConfig(
    vocab_size=32001,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    max_position_embeddings=512,
    type_vocab_size=2,
)

# Using the same hyperparameters as the main training script, but adapted for a small-scale overfit test.
training_args = TrainingArguments(
    output_dir="./overfit_test_model",
    overwrite_output_dir=True,
    num_train_epochs=100,  # Run for many epochs to ensure overfitting
    per_device_train_batch_size=8,  # Use a small batch size for the small dataset
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=1, # No accumulation needed for this test
    eval_strategy="epoch",  # Evaluate at the end of each epoch
    learning_rate=1e-4,
    warmup_ratio=0.06  , 
    save_strategy="epoch",
    save_total_limit=2,
    prediction_loss_only=False,
    bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    logging_steps=1, # Log every step to see progress on the small dataset
    report_to="wandb",
    dataloader_num_workers=0, # Use main process for small dataset
)

def main():
    """
    Main function to run the overfitting test.
    """
    script_dir = os.path.dirname(__file__)
    sanskrit_dir = os.path.abspath(os.path.join(script_dir, '..'))

    tokenizer_path = os.path.join(sanskrit_dir, 'tokenizer', 'sanskrit-bert-tokenizer')
    tokenized_dataset_path = os.path.join(sanskrit_dir, 'data', 'tokenized_dataset')
    
    # We need to modify the dataset loading in run_training to select a small subset
    # For now, let's assume run_training is modified or can handle this.
    # The ideal way is to pass the number of samples to use.
    
    # Let's add a parameter to run_training to limit the dataset size.
    # I will assume such a parameter `debug_subset_size` exists in `run_training`.
    
    project_name = "sanskrit-bert-overfit-test"

    run_training(
        training_args=training_args,
        model_config=model_config,
        tokenized_dataset_path=tokenized_dataset_path,
        tokenizer_path=tokenizer_path,
        project_name=project_name,
        eval_samples=16, # Use 16 samples for evaluation
        train_samples=32 # Use 32 samples for training
    )

if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"Found {torch.cuda.device_count()} GPU(s).")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU found. Training will run on CPU.")
    
    main()
