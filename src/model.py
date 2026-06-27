from unsloth import FastLanguageModel
import torch

def initialize_model(config):
    """
    Loads base 4-bit quantized Qwen2.5-Coder via FastLanguageModel and applies PEFT/QLoRA adapters.
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model.name_or_path,
        max_seq_length=config.model.max_seq_length,
        dtype=None, 
        load_in_4bit=config.model.load_in_4bit,
    )

    # Apply PEFT / QLoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora.r,
        target_modules=config.lora.target_modules,
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.lora_dropout,
        bias=config.lora.bias,
        use_gradient_checkpointing="unsloth",
        random_state=config.training.seed,
        use_rslora=False,
        loftq_config=None,
    )
    
    return model, tokenizer
