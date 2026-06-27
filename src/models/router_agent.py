from unsloth import FastLanguageModel
import torch
from src.core.config import AppConfig

class RouterAgent:
    def __init__(self, config: AppConfig):
        self.config = config
        
    def initialize_model(self):
        """Loads base model and applies LoRA adapters."""
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.model.name_or_path,
            max_seq_length=self.config.model.max_seq_length,
            dtype=None, 
            load_in_4bit=self.config.model.load_in_4bit,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=self.config.lora.r,
            target_modules=self.config.lora.target_modules,
            lora_alpha=self.config.lora.lora_alpha,
            lora_dropout=self.config.lora.lora_dropout,
            bias=self.config.lora.bias,
            use_gradient_checkpointing="unsloth",
            random_state=self.config.training.seed,
            use_rslora=False,
            loftq_config=None,
        )
        
        return model, tokenizer
        
    def load_for_inference(self, lora_path: str):
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=lora_path, 
            max_seq_length=self.config.model.max_seq_length,
            dtype=None,
            load_in_4bit=self.config.model.load_in_4bit,
        )
        FastLanguageModel.for_inference(model)
        return model, tokenizer
