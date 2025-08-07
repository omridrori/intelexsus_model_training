import os
# ✨ Add the following two lines right here at the beginning ✨
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from pathlib import Path
from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertTokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset
import torch
from tqdm import tqdm
import logging

# --- 1. Main Settings (No Changes) ---
TOKENIZER_PATH = Path("sanskrit-bert-tokenizer")
DATA_FILE = Path("data/sanskrit_preprocessed/sanskrit_bert_ready_512.txt")
MODEL_OUTPUT_DIR = "sanskrit-bert-from-scratch"
WANDB_PROJECT_NAME = "Sanskrit-BERT-Pretraining"

# --- 2. Model Configuration (No Changes) ---
model_config = BertConfig(
    vocab_size=32000,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    max_position_embeddings=512,
)

# --- 3. Training Settings (Optimized for 12GB GPU with Resume Capability) ---
training_args = TrainingArguments(
    output_dir=MODEL_OUTPUT_DIR,
    overwrite_output_dir=False,  # ✨ Important fix: allows resuming training
    num_train_epochs=5,
    
    # --- ✨ Critical Memory Saving Changes ---
    per_device_train_batch_size=2,          # Reduced physical batch size
    gradient_accumulation_steps=16,         # Increased to compensate (effective batch of 32)
    gradient_checkpointing=True,            # Important trick for huge memory savings
    # ---------------------------------------------

    fp16=True,                            # Essential for memory savings and speed
    save_steps=100,                    # Saves checkpoint every 10,000 steps
    save_total_limit=3,
    prediction_loss_only=True,
    logging_steps=20,
    report_to="wandb",
    
    # Additional optimizations
    dataloader_num_workers=4,
    warmup_ratio=0.06,
    learning_rate=1e-4,
    weight_decay=0.01,
    max_grad_norm=1.0,
)

# --- 4. Main Training Function (No Changes) ---
def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    os.environ["WANDB_PROJECT"] = WANDB_PROJECT_NAME

    print("=== Sanskrit BERT Training Setup ===")
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"Found {torch.cuda.device_count()} GPU(s).")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("No GPU found. Training will run on CPU, which might be very slow.")

    print(f"\nLoading tokenizer from {TOKENIZER_PATH}...")
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(f"Tokenizer not found at {TOKENIZER_PATH}")
    
    tokenizer = BertTokenizerFast.from_pretrained(str(TOKENIZER_PATH))
    print(f"Tokenizer loaded successfully. Vocabulary size: {tokenizer.vocab_size}")
    
    print("\nInitializing a new BERT model from scratch...")
    model = BertForMaskedLM(config=model_config)
    print(f"Model created with {model.num_parameters():,} parameters.")

    print(f"\nLoading and tokenizing dataset from {DATA_FILE}...")
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Data file not found at {DATA_FILE}")
    
    dataset = load_dataset("text", data_files={"train": str(DATA_FILE)})
    print(f"Dataset loaded with {len(dataset['train'])} examples")

    def tokenize_function(examples):
        """Tokenize the text with progress tracking"""
        return tokenizer(
            examples["text"], 
            truncation=True, 
            max_length=512, 
            padding="max_length"
        )

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function, 
        batched=True, 
        num_proc=4,
        remove_columns=["text"],
        desc="Tokenizing dataset"
    )
    
    print("Dataset tokenized and ready.")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
    )

    print("\n=== Starting Model Training ===")
    print(f"Training for {training_args.num_train_epochs} epochs")
    print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"Steps per epoch: {len(tokenized_dataset['train']) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)}")
    
    trainer.train()
    print("=== Training Finished ===")
    
    final_model_path = Path(MODEL_OUTPUT_DIR) / "final_model"
    print(f"\nSaving final model and tokenizer to {final_model_path}")
    trainer.save_model(str(final_model_path))
    tokenizer.save_pretrained(str(final_model_path))
    print(f"Final model and tokenizer saved to {final_model_path}")
    
    # Print training summary
    print("\n=== Training Summary ===")
    print(f"Model saved to: {final_model_path}")
    print(f"Total parameters: {model.num_parameters():,}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print("Training completed successfully!")


if __name__ == "__main__":
    main() 