import yaml
from dataclasses import dataclass
from typing import List

@dataclass
class ModelConfig:
    name_or_path: str
    max_seq_length: int
    load_in_4bit: bool

@dataclass
class LoraConfig:
    r: int
    lora_alpha: int
    lora_dropout: float
    bias: str
    target_modules: List[str]

@dataclass
class DataConfig:
    dataset_name: str

@dataclass
class TrainingConfig:
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    warmup_steps: int
    max_steps: int
    learning_rate: float
    logging_steps: int
    optim: str
    weight_decay: float
    lr_scheduler_type: str
    seed: int
    output_dir: str

@dataclass
class AppConfig:
    model: ModelConfig
    lora: LoraConfig
    data: DataConfig
    training: TrainingConfig

def load_config(config_path: str) -> AppConfig:
    """Parses, type-validates, and loads config.yaml properties."""
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
    
    return AppConfig(
        model=ModelConfig(**data['model']),
        lora=LoraConfig(**data['lora']),
        data=DataConfig(**data['data']),
        training=TrainingConfig(
            per_device_train_batch_size=data['training']['per_device_train_batch_size'],
            gradient_accumulation_steps=data['training']['gradient_accumulation_steps'],
            warmup_steps=data['training']['warmup_steps'],
            max_steps=data['training']['max_steps'],
            learning_rate=float(data['training']['learning_rate']),
            logging_steps=data['training']['logging_steps'],
            optim=data['training']['optim'],
            weight_decay=float(data['training']['weight_decay']),
            lr_scheduler_type=data['training']['lr_scheduler_type'],
            seed=data['training']['seed'],
            output_dir=data['training']['output_dir']
        )
    )
