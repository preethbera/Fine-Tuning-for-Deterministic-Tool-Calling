import argparse
import os
from src.config_loader import load_config
from src.utils import enforce_gc
from src.data_processing import prepare_dataset
from src.model import initialize_model
from src.trainer import setup_and_train
from src.inference import generate_response, validate_structure

def main():
    parser = argparse.ArgumentParser(description="Deterministic Agentic Tool Router Pipeline")
    parser.add_argument('--phase', type=str, choices=['process', 'train', 'test', 'all'], default='all',
                        help='Which phase of the pipeline to run')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config yaml file')
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    model = None
    tokenizer = None
    dataset = None
    
    # Process Phase: Load data and model tokenizer for formatting
    if args.phase in ['process', 'train', 'all']:
        print("=== Phase: Process ===")
        # Unsloth models define the tokenizer setup. We need it to format the dataset natively.
        model, tokenizer = initialize_model(config)
        dataset = prepare_dataset(config.data.dataset_name, tokenizer)
        print(f"Dataset prepared with {len(dataset)} training examples.")
        enforce_gc()
        
    # Train Phase: Execute fine-tuning
    if args.phase in ['train', 'all']:
        print("\n=== Phase: Train ===")
        setup_and_train(model, tokenizer, dataset, config)
        print(f"Training complete. Adapters saved to {config.training.output_dir}/lora_model")
        enforce_gc()
        
    # Test Phase: Structural validation against a mock tool registry
    if args.phase in ['test', 'all']:
        print("\n=== Phase: Test ===")
        if model is None or tokenizer is None:
            # If skipping straight to test, we need to load the trained model
            lora_path = os.path.join(config.training.output_dir, "lora_model")
            if not os.path.exists(lora_path):
                print(f"Error: Model not found at {lora_path}. Please train first.")
                return
                
            from unsloth import FastLanguageModel
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=lora_path, 
                max_seq_length=config.model.max_seq_length,
                dtype=None,
                load_in_4bit=config.model.load_in_4bit,
            )
            
        mock_tools = [{
            "name": "get_weather",
            "description": "Get current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City and state, e.g. San Francisco, CA"}
                },
                "required": ["location"]
            }
        }]
        
        # Test 1: Valid Scenario
        print("\n[Test Case 1: Valid intent matching]")
        response1 = generate_response(model, tokenizer, "What's the weather like in London today?", mock_tools)
        validate_structure(response1)
        
        # Test 2: Invalid / Abort Scenario
        print("\n[Test Case 2: No matching tool (ABORT scenario)]")
        response2 = generate_response(model, tokenizer, "Can you set a timer for 10 minutes?", mock_tools)
        validate_structure(response2)
        
        # Test 3: Invalid / Abort Scenario (Missing parameter)
        print("\n[Test Case 3: Missing parameter (ABORT scenario)]")
        response3 = generate_response(model, tokenizer, "What's the weather like?", mock_tools)
        validate_structure(response3)

if __name__ == "__main__":
    main()
