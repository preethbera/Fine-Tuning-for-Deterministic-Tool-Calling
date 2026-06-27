import os
import yaml
from typing import List, Optional
from pydantic import BaseModel

class ModelConfig(BaseModel):
    name_or_path: str
    max_seq_length: int
    load_in_4bit: bool

class LoraConfig(BaseModel):
    r: int
    lora_alpha: int
    lora_dropout: float
    bias: str
    target_modules: List[str]

class DataConfig(BaseModel):
    dataset_name: str
    dataset_limit: Optional[int] = None

class TrainingConfig(BaseModel):
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    warmup_steps: int
    max_steps: int
    learning_rate: float
    logging_steps: int
    save_steps: int
    optim: str
    weight_decay: float
    lr_scheduler_type: str
    seed: int
    output_dir: str
    logging_dir: str

class AppConfig(BaseModel):
    model: ModelConfig
    lora: LoraConfig
    data: DataConfig
    training: TrainingConfig

def deep_merge(dict1: dict, dict2: dict) -> dict:
    """Recursively merges dict2 into dict1."""
    merged = dict1.copy()
    for key, value in dict2.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged

def load_config(profile: str = "base") -> AppConfig:
    """Loads configuration, merging smoke_test on top of base if profile is smoke_test."""
    base_path = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "base.yaml")
    
    with open(base_path, "r") as f:
        config_dict = yaml.safe_load(f)
        
    if profile == "smoke_test":
        smoke_path = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "smoke_test.yaml")
        with open(smoke_path, "r") as f:
            smoke_dict = yaml.safe_load(f)
            config_dict = deep_merge(config_dict, smoke_dict)
            
    return AppConfig(**config_dict)
