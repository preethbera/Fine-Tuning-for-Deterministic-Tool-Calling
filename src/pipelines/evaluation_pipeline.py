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
        
    def _is_dict_match(self, predicted, ground_truth):
        """Recursively checks if two dictionaries match exactly in keys and values."""
        if isinstance(predicted, dict) and isinstance(ground_truth, dict):
            if set(predicted.keys()) != set(ground_truth.keys()):
                return False
            for k in predicted:
                if not self._is_dict_match(predicted[k], ground_truth[k]):
                    return False
            return True
        elif isinstance(predicted, list) and isinstance(ground_truth, list):
            if len(predicted) != len(ground_truth):
                return False
            for p, g in zip(predicted, ground_truth):
                if not self._is_dict_match(p, g):
                    return False
            return True
        else:
            return predicted == ground_truth
            
    def _is_type_match(self, predicted, ground_truth):
        """Recursively checks if values in predicted match the types in ground_truth."""
        if type(predicted) != type(ground_truth):
            # Special case for numeric types that might be interchangeable in JSON
            if isinstance(predicted, (int, float)) and isinstance(ground_truth, (int, float)):
                pass
            else:
                return False
                
        if isinstance(predicted, dict) and isinstance(ground_truth, dict):
            for k in predicted:
                if k in ground_truth:
                    if not self._is_type_match(predicted[k], ground_truth[k]):
                        return False
            return True
        elif isinstance(predicted, list) and isinstance(ground_truth, list):
            for p, g in zip(predicted, ground_truth):
                if not self._is_type_match(p, g):
                    return False
            return True
        return True

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
            "total_positive_cases": 0,
            "total_negative_cases": 0,
            "ast_match_count": 0,
            "type_fidelity_count": 0,
            "negative_rejection_count": 0,
            "hallucination_count": 0
        }
        
        from src.pipelines.data_pipeline import SYSTEM_PROMPT
        
        for item in tqdm(dataset, desc="Evaluating"):
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
            available_tool_names = [t.get("name") for t in tools_json] if isinstance(tools_json, list) else []
            
            predicted_tools = []
            if parsed["is_valid_json"] and parsed["parsed_json"]:
                if isinstance(parsed["parsed_json"], list):
                    predicted_tools = parsed["parsed_json"]
                elif isinstance(parsed["parsed_json"], dict):
                    predicted_tools = [parsed["parsed_json"]]
                    
            predicted_tool_names = [pt.get("name") for pt in predicted_tools if isinstance(pt, dict)]
            
            for pt_name in predicted_tool_names:
                if pt_name not in available_tool_names:
                    metrics["hallucination_count"] += 1
                    break

            if not answers_json:
                metrics["total_negative_cases"] += 1
                if parsed["is_abort"]:
                    metrics["negative_rejection_count"] += 1
            else:
                metrics["total_positive_cases"] += 1
                
                if parsed["is_valid_json"] and len(predicted_tools) > 0 and len(answers_json) > 0:
                    pt = predicted_tools[0]
                    gt = answers_json[0]
                    
                    if isinstance(pt, dict) and isinstance(gt, dict):
                        pt_name = str(pt.get("name", "")).strip()
                        gt_name = str(gt.get("name", "")).strip()
                        if pt_name and pt_name == gt_name:
                            pt_args = pt.get("arguments", {})
                            gt_args = gt.get("arguments", {})
                            
                            if isinstance(pt_args, str):
                                try:
                                    pt_args = json.loads(pt_args)
                                except:
                                    pt_args = {}
                            if isinstance(gt_args, str):
                                try:
                                    gt_args = json.loads(gt_args)
                                except:
                                    gt_args = {}
                                    
                            if self._is_dict_match(pt_args, gt_args):
                                metrics["ast_match_count"] += 1
                                
                            if set(pt_args.keys()) == set(gt_args.keys()):
                                if self._is_type_match(pt_args, gt_args):
                                    metrics["type_fidelity_count"] += 1

        report = {
            "total_samples": metrics["total_samples"],
            "ast_match_rate": metrics["ast_match_count"] / max(1, metrics["total_positive_cases"]),
            "argument_type_fidelity": metrics["type_fidelity_count"] / max(1, metrics["total_positive_cases"]),
            "negative_rejection_accuracy": metrics["negative_rejection_count"] / max(1, metrics["total_negative_cases"]),
            "hallucination_rate": metrics["hallucination_count"] / max(1, metrics["total_samples"])
        }
        
        with open("bfcl_evaluation_metrics.json", "w") as f:
            json.dump(report, f, indent=4)
            
        print("Evaluation complete. Results saved to bfcl_evaluation_metrics.json")
        return report
