from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from src.utils import ensure_dir
import os

def setup_and_train(model, tokenizer, dataset, config):
    """
    Core training workflow using SFTTrainer with metrics and checkpointing.
    """
    ensure_dir(config.training.output_dir)
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=config.model.max_seq_length,
        dataset_num_proc=2,
        packing=False, # Set to False for function calling data to avoid cross-contamination
        args=TrainingArguments(
            per_device_train_batch_size=config.training.per_device_train_batch_size,
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            warmup_steps=config.training.warmup_steps,
            max_steps=config.training.max_steps,
            learning_rate=config.training.learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=config.training.logging_steps,
            optim=config.training.optim,
            weight_decay=config.training.weight_decay,
            lr_scheduler_type=config.training.lr_scheduler_type,
            seed=config.training.seed,
            output_dir=config.training.output_dir,
            report_to="none" # Disable wandb for basic Kaggle run, can be enabled via config later
        ),
    )
    
    # Train the model
    trainer_stats = trainer.train()
    
    # Save model adapters and tokenizer
    save_path = os.path.join(config.training.output_dir, "lora_model")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    return trainer_stats
