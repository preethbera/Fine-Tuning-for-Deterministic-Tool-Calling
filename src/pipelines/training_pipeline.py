import unsloth
import os
import math
from datasets import load_from_disk
from unsloth import is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback, TrainerState, TrainerControl
from transformers.trainer_utils import get_last_checkpoint
from src.core.config import AppConfig
from src.models.router_agent import RouterAgent
from src.core.exceptions import DivergenceException

class DivergenceGuard(TrainerCallback):
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            loss = logs["loss"]
            if math.isnan(loss) or math.isinf(loss):
                print(f"\n[DIVERGENCE GUARD TRIGGERED] Invalid loss detected: {loss}.")
                control.should_training_stop = True
                raise DivergenceException(f"Training loss diverged to {loss}")

class TrainingPipeline:
    def __init__(self, config: AppConfig):
        self.config = config
        
    def execute(self):
        try:
            dataset = load_from_disk("./data/processed_dataset")
        except Exception as e:
            raise Exception("Could not load processed dataset. Run data pipeline first.") from e
            
        os.makedirs(self.config.training.output_dir, exist_ok=True)
        os.makedirs(self.config.training.logging_dir, exist_ok=True)
        os.environ["TENSORBOARD_LOGGING_DIR"] = self.config.training.logging_dir
        
        agent = RouterAgent(self.config)
        model, tokenizer = agent.initialize_model()
        
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=self.config.model.max_seq_length,
            dataset_num_proc=2,
            packing=False,
            args=TrainingArguments(
                per_device_train_batch_size=self.config.training.per_device_train_batch_size,
                gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
                warmup_steps=self.config.training.warmup_steps,
                max_steps=self.config.training.max_steps,
                learning_rate=self.config.training.learning_rate,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=self.config.training.logging_steps,
                save_steps=self.config.training.save_steps,
                optim=self.config.training.optim,
                weight_decay=self.config.training.weight_decay,
                lr_scheduler_type=self.config.training.lr_scheduler_type,
                seed=self.config.training.seed,
                output_dir=self.config.training.output_dir,
                report_to="tensorboard"
            ),
            callbacks=[DivergenceGuard()]
        )
        
        last_checkpoint = None
        if os.path.isdir(self.config.training.output_dir):
            last_checkpoint = get_last_checkpoint(self.config.training.output_dir)
            
        try:
            trainer.train(resume_from_checkpoint=last_checkpoint)
        except DivergenceException as e:
            print(f"Training aborted safely: {e}")
            
        save_path = os.path.join(self.config.training.output_dir, "lora_model")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"Training complete. Model saved to {save_path}")
