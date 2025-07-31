import os
from transformers import BertConfig, TrainingArguments
import torch
from training_utils import run_training

# --- Configuration ---
model_config = BertConfig(
    vocab_size=32001,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    max_position_embeddings=512,
    type_vocab_size=2,
)

training_args = TrainingArguments(
    output_dir="./sanskrit_bert_model",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=3,  # Increased for potentially faster processing
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,  # Adjusted to keep effective batch size at 32
    eval_strategy="steps",
    eval_steps=10000,  # Evaluate every 500 steps
    learning_rate=1.5e-5,  # Lowered learning rate for more stable training
    weight_decay=0.01,  # Added weight decay for regularization
    warmup_ratio=0.06,                 # Use ratio instead of a fixed number of steps
    eval_accumulation_steps=5,  # Explicitly set for evaluation
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=False,  # Set to False to compute metrics
    bf16=True,
    logging_steps=8,
    report_to="wandb",
    dataloader_num_workers=6,
)

def main():
    """
    Main function to set up and run the training process.
    """
    script_dir = os.path.dirname(__file__)
    sanskrit_dir = os.path.abspath(os.path.join(script_dir, '..'))

    tokenizer_path = os.path.join(sanskrit_dir, 'tokenizer', 'sanskrit-bert-tokenizer')

    tokenized_dataset_path = os.path.join(sanskrit_dir, 'data', 'tokenized_dataset')
    project_name = "sanskrit-bert-training"

    run_training(
        training_args=training_args,
        model_config=model_config,
        tokenized_dataset_path=tokenized_dataset_path,
        tokenizer_path=tokenizer_path,
        project_name=project_name,
        eval_samples=1600 # Use a small, fixed number of samples for evaluation
    )

if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"Found {torch.cuda.device_count()} GPU(s).")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU found. Training will run on CPU, which might be very slow.")
    
    main() 