import unsloth
import os
import math

try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    os.environ["WANDB_API_KEY"] = user_secrets.get_secret("WANDB_API_KEY")
    os.environ["WANDB_PROJECT"] = "agentic-tool-router"
    print("Successfully loaded WANDB_API_KEY from Kaggle secrets.")
except ImportError:
    print("Warning: kaggle_secrets not found. If running locally, ensure WANDB_API_KEY is set in environment.")
except Exception as e:
    print(f"Warning: Failed to load WANDB_API_KEY from Kaggle secrets: {e}")

from datasets import load_from_disk
from unsloth import is_bfloat16_supported
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback, TrainerState, TrainerControl
from transformers.trainer_utils import get_last_checkpoint
from src.core.config import AppConfig
from src.models.router_agent import RouterAgent
from src.core.exceptions import DivergenceException

class DivergenceGuard(TrainerCallback):
    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
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
            args=SFTConfig(
                dataset_text_field="text",
                max_seq_length=self.config.model.max_seq_length,
                dataset_num_proc=2,
                packing=False,
                per_device_train_batch_size=self.config.training.per_device_train_batch_size,
                gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
                warmup_steps=self.config.training.warmup_steps,
                num_train_epochs=self.config.training.num_train_epochs,
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
                report_to=self.config.training.report_to,
                run_name=self.config.training.run_name
            ),
            callbacks=[DivergenceGuard()]
        )
        
        last_checkpoint = None
        if os.path.isdir(self.config.training.output_dir):
            last_checkpoint = get_last_checkpoint(self.config.training.output_dir)
            if last_checkpoint and not os.path.exists(os.path.join(last_checkpoint, "trainer_state.json")):
                print(f"Warning: Corrupted checkpoint found at {last_checkpoint} (missing trainer_state.json). Starting fresh.")
                last_checkpoint = None
            
        try:
            trainer.train(resume_from_checkpoint=last_checkpoint)
        except DivergenceException as e:
            print(f"Training aborted safely: {e}")
            
        save_path = os.path.join(self.config.training.output_dir, "lora_model")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"Training complete. Model saved to {save_path}")
