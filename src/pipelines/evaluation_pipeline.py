import os
import json
import torch
from datasets import load_dataset
from tqdm import tqdm
from src.core.config import AppConfig
from src.models.router_agent import RouterAgent
from src.models.validator import Validator
from src.adapters.base_parser import BaseDatasetParser

class EvaluationPipeline:
    def __init__(self, config: AppConfig, parser: BaseDatasetParser):
        self.config = config
        self.parser = parser
        
    def execute(self, num_samples: int = 100):
        agent = RouterAgent(self.config)
        lora_path = os.path.join(self.config.training.output_dir, "lora_model")
        
        if not os.path.exists(lora_path):
            print(f"Model not found at {lora_path}. Please train first.")
            return
            
        model, tokenizer = agent.load_for_inference(lora_path)
        
        dataset_name = self.config.data.dataset_name
        try:
            dataset = load_dataset(dataset_name, split="test")
        except ValueError:
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
        
        from src.pipelines.data_pipeline import SYSTEM_PROMPT
        
        for item in tqdm(dataset):
            try:
                record = self.parser.transform(item)
            except Exception:
                continue
                
            query = record.query
            tools_json = record.tools
            answers_str = record.answers
            
            try:
                answers_json = json.loads(answers_str) if answers_str and answers_str != "[]" else []
            except:
                answers_json = []
                
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
            
            attention_mask = torch.ones_like(inputs)
            
            outputs = model.generate(
                input_ids=inputs, 
                attention_mask=attention_mask,
                max_new_tokens=512, 
                max_length=None,
                use_cache=True, 
                pad_token_id=tokenizer.eos_token_id
            )
            
            input_length = inputs.shape[1]
            response_tokens = outputs[0][input_length:]
            response = tokenizer.decode(response_tokens, skip_special_tokens=True)
            
            parsed = Validator.parse_generated_text(response)
            
            metrics["total_samples"] += 1
            
            if parsed["has_thought"]:
                if parsed["is_abort"] and not parsed["has_tool_call"]:
                    metrics["format_adherence"] += 1
                elif not parsed["is_abort"] and parsed["has_tool_call"]:
                    metrics["format_adherence"] += 1
                    
            if not answers_json:
                metrics["total_negative_cases"] += 1
                if parsed["is_abort"]:
                    metrics["negative_trigger_success"] += 1
            else:
                metrics["total_positive_cases"] += 1
                if parsed["has_tool_call"] and parsed["is_valid_json"]:
                    metrics["json_parse_success"] += 1
                    
                    predicted_tools = []
                    if isinstance(parsed["parsed_json"], list):
                        predicted_tools = [t.get("name") for t in parsed["parsed_json"] if isinstance(t, dict)]
                    elif isinstance(parsed["parsed_json"], dict):
                        predicted_tools = [parsed["parsed_json"].get("name")]
                    
                    ground_truth_tools = [a.get("name") for a in answers_json] if isinstance(answers_json, list) else []
                    
                    if predicted_tools and ground_truth_tools and predicted_tools[0] == ground_truth_tools[0]:
                        metrics["tool_selection_accuracy"] += 1
                        
        report = {
            "total_samples": metrics["total_samples"],
            "format_adherence_rate": metrics["format_adherence"] / max(1, metrics["total_samples"]),
            "json_parse_success_rate": metrics["json_parse_success"] / max(1, metrics["total_positive_cases"]) if metrics["total_positive_cases"] > 0 else 0.0,
            "tool_selection_accuracy": metrics["tool_selection_accuracy"] / max(1, metrics["total_positive_cases"]) if metrics["total_positive_cases"] > 0 else 0.0,
            "negative_trigger_success_rate": metrics["negative_trigger_success"] / max(1, metrics["total_negative_cases"]) if metrics["total_negative_cases"] > 0 else 1.0,
        }
        
        with open("eval_results.json", "w") as f:
            json.dump(report, f, indent=4)
            
        print("Evaluation complete. Results saved to eval_results.json")
        return report
