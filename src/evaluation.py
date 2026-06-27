import json
import re
import os
from tqdm import tqdm
from datasets import load_dataset
from unsloth import FastLanguageModel

def parse_generated_text(text: str):
    """
    Parses the generated output text for validation testing.
    """
    has_thought = bool(re.search(r"<thought>.*?</thought>", text, re.DOTALL))
    
    tool_match = re.search(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
    has_tool_call = bool(tool_match)
    
    is_valid_json = False
    parsed_json = None
    if has_tool_call:
        try:
            parsed_json = json.loads(tool_match.group(1).strip())
            is_valid_json = True
        except json.JSONDecodeError:
            pass
            
    is_abort = "[ABORT:" in text
    
    return {
        "has_thought": has_thought,
        "has_tool_call": has_tool_call,
        "is_valid_json": is_valid_json,
        "parsed_json": parsed_json,
        "is_abort": is_abort
    }

def run_evaluation(model, tokenizer, dataset_name: str, num_samples: int = 100):
    """
    Runs deterministic evaluation loop and calculates performance metrics.
    """
    FastLanguageModel.for_inference(model)
    from src.data_processing import SYSTEM_PROMPT
    
    print(f"Loading evaluation split from {dataset_name}...")
    try:
        dataset = load_dataset(dataset_name, split="test")
    except ValueError:
        print(f"No test split found, sampling from train split.")
        dataset = load_dataset(dataset_name, split="train")
        dataset = dataset.select(range(len(dataset)-num_samples, len(dataset)))
        
    dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    metrics = {
        "total_samples": 0,
        "format_adherence": 0,
        "json_parse_success": 0,
        "tool_selection_accuracy": 0,
        "negative_trigger_success": 0,
        "total_negative_cases": 0,
        "total_positive_cases": 0
    }
    
    print(f"Running evaluation loop over {len(dataset)} samples...")
    for item in tqdm(dataset):
        query = item['query']
        tools_json = json.loads(item['tools']) if isinstance(item['tools'], str) else item['tools']
        answers_json = json.loads(item['answers']) if isinstance(item['answers'], str) else item['answers']
        
        tools_str = json.dumps(tools_json, indent=2)
        user_message = f"Available tools:\n{tools_str}\n\nUser Query: {query}"
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]
        
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")
        
        outputs = model.generate(
            input_ids=inputs, 
            max_new_tokens=512, 
            use_cache=True, 
            pad_token_id=tokenizer.eos_token_id
        )
        
        input_length = inputs.shape[1]
        response_tokens = outputs[0][input_length:]
        response = tokenizer.decode(response_tokens, skip_special_tokens=True)
        
        parsed = parse_generated_text(response)
        
        metrics["total_samples"] += 1
        
        # 1. Format Adherence Rate
        if parsed["has_thought"]:
            if parsed["is_abort"] and not parsed["has_tool_call"]:
                metrics["format_adherence"] += 1
            elif not parsed["is_abort"] and parsed["has_tool_call"]:
                metrics["format_adherence"] += 1
                
        # Handle ground truth categorization
        if not answers_json:
            # Negative Case
            metrics["total_negative_cases"] += 1
            if parsed["is_abort"]:
                metrics["negative_trigger_success"] += 1
        else:
            # Positive Case
            metrics["total_positive_cases"] += 1
            
            # 2. JSON Parse Success Rate
            if parsed["has_tool_call"] and parsed["is_valid_json"]:
                metrics["json_parse_success"] += 1
                
                # 3. Tool Selection Accuracy
                predicted_tools = []
                if isinstance(parsed["parsed_json"], list):
                    predicted_tools = [t.get("name") for t in parsed["parsed_json"] if isinstance(t, dict)]
                elif isinstance(parsed["parsed_json"], dict):
                    predicted_tools = [parsed["parsed_json"].get("name")]
                
                ground_truth_tools = [a.get("name") for a in answers_json]
                
                if predicted_tools and ground_truth_tools and predicted_tools[0] == ground_truth_tools[0]:
                    metrics["tool_selection_accuracy"] += 1
                    
    report = {
        "total_samples": metrics["total_samples"],
        "format_adherence_rate": metrics["format_adherence"] / max(1, metrics["total_samples"]),
        "json_parse_success_rate": metrics["json_parse_success"] / max(1, metrics["total_positive_cases"]) if metrics["total_positive_cases"] > 0 else 0,
        "tool_selection_accuracy": metrics["tool_selection_accuracy"] / max(1, metrics["total_positive_cases"]) if metrics["total_positive_cases"] > 0 else 0,
        "negative_trigger_success_rate": metrics["negative_trigger_success"] / max(1, metrics["total_negative_cases"]) if metrics["total_negative_cases"] > 0 else 1.0,
        "total_negative_cases": metrics["total_negative_cases"],
        "total_positive_cases": metrics["total_positive_cases"]
    }
    
    with open("eval_results.json", "w") as f:
        json.dump(report, f, indent=4)
        
    return report
