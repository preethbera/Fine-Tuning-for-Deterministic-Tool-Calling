import os
import json
import time
import torch
import traceback
import matplotlib.pyplot as plt
import seaborn as sns
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
        if type(predicted) != type(ground_truth):
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
        
        reports_dir = os.path.join(self.config.training.output_dir, "evaluation_reports")
        os.makedirs(reports_dir, exist_ok=True)
        
        trace_path = os.path.join(reports_dir, "evaluation_debug_trace.jsonl")
        
        metrics = {
            "total_samples": 0,
            "true_positives": 0,
            "false_positives": 0,
            "true_negatives": 0,
            "false_negatives": 0,
            
            # Error distributions
            "error_syntax": 0,
            "error_hallucinated_tool": 0,
            "error_hallucinated_param": 0,
            "error_missing_param": 0,
            "error_type_mismatch": 0,
            "error_value_mismatch": 0,
            "error_missing_tool": 0,
            
            # Performance
            "total_inference_latency_ms": 0,
            "total_generated_tokens": 0
        }
        
        from src.pipelines.data_pipeline import SYSTEM_PROMPT
        
        with open(trace_path, "w") as f_trace:
            for item in tqdm(dataset, desc="Evaluating"):
                try:
                    record = self.parser.transform(item)
                except Exception:
                    continue
                    
                query = record.query
                tools_json = record.tools
                answers_str = record.answers
                
                ground_truth_parsing_error = False
                try:
                    answers_json = json.loads(answers_str) if answers_str and answers_str != "[]" else []
                except:
                    answers_json = []
                    ground_truth_parsing_error = True
                    
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
                
                start_time = time.time()
                outputs = model.generate(
                    input_ids=inputs, 
                    attention_mask=attention_mask,
                    max_new_tokens=512, 
                    max_length=None,
                    use_cache=True, 
                    pad_token_id=tokenizer.eos_token_id
                )
                end_time = time.time()
                
                latency_ms = (end_time - start_time) * 1000
                input_length = inputs.shape[1]
                response_tokens = outputs[0][input_length:]
                num_generated_tokens = len(response_tokens)
                
                metrics["total_inference_latency_ms"] += latency_ms
                metrics["total_generated_tokens"] += num_generated_tokens
                
                response = tokenizer.decode(response_tokens, skip_special_tokens=True)
                parsed = Validator.parse_generated_text(response)
                
                metrics["total_samples"] += 1
                available_tool_names = []
                if isinstance(tools_json, list):
                    for t in tools_json:
                        if isinstance(t, dict):
                            if "function" in t and isinstance(t["function"], dict):
                                available_tool_names.append(t["function"].get("name"))
                            else:
                                available_tool_names.append(t.get("name"))
                
                predicted_tools = []
                if parsed["is_valid_json"] and parsed["parsed_json"]:
                    if isinstance(parsed["parsed_json"], list):
                        predicted_tools = parsed["parsed_json"]
                    elif isinstance(parsed["parsed_json"], dict):
                        predicted_tools = [parsed["parsed_json"]]
                        
                outcome = "Unknown"
                error_category = None
                
                # Evaluation Logic
                if not answers_json:
                    # Expecting Negative (Abort)
                    if parsed["is_abort"] or not predicted_tools:
                        metrics["true_negatives"] += 1
                        outcome = "True Negative"
                    else:
                        metrics["false_positives"] += 1
                        outcome = "False Positive"
                        error_category = "error_hallucinated_tool"
                        metrics[error_category] += 1
                else:
                    # Expecting Positive Tool Call
                    if parsed["is_abort"] or not predicted_tools:
                        metrics["false_negatives"] += 1
                        outcome = "False Negative"
                        if not parsed["is_valid_json"] and parsed["has_tool_call"]:
                            error_category = "error_syntax"
                            metrics[error_category] += 1
                        else:
                            error_category = "error_missing_tool" # model refused or failed completely
                    else:
                        pt = predicted_tools[0]
                        gt = answers_json[0]
                        
                        if not isinstance(pt, dict):
                            metrics["false_negatives"] += 1
                            outcome = "False Negative"
                            error_category = "error_syntax"
                            metrics[error_category] += 1
                        else:
                            pt_name = str(pt.get("name", "")).strip()
                            gt_name = str(gt.get("name", "")).strip()
                            
                            if pt_name not in available_tool_names:
                                metrics["false_negatives"] += 1
                                outcome = "False Negative"
                                error_category = "error_hallucinated_tool"
                                metrics[error_category] += 1
                            elif pt_name != gt_name:
                                metrics["false_negatives"] += 1
                                outcome = "False Negative"
                                error_category = "error_hallucinated_tool"
                                metrics[error_category] += 1
                            else:
                                pt_args = pt.get("arguments", {})
                                gt_args = gt.get("arguments", {})
                                
                                if isinstance(pt_args, str):
                                    try: pt_args = json.loads(pt_args)
                                    except: pt_args = {}
                                if isinstance(gt_args, str):
                                    try: gt_args = json.loads(gt_args)
                                    except: gt_args = {}
                                        
                                if self._is_dict_match(pt_args, gt_args):
                                    metrics["true_positives"] += 1
                                    outcome = "True Positive"
                                else:
                                    metrics["false_negatives"] += 1
                                    outcome = "False Negative"
                                    
                                    pt_keys = set(pt_args.keys())
                                    gt_keys = set(gt_args.keys())
                                    
                                    if pt_keys - gt_keys:
                                        error_category = "error_hallucinated_param"
                                    elif gt_keys - pt_keys:
                                        error_category = "error_missing_param"
                                    elif not self._is_type_match(pt_args, gt_args):
                                        error_category = "error_type_mismatch"
                                    else:
                                        error_category = "error_value_mismatch"
                                        
                                    if error_category in metrics:
                                        metrics[error_category] += 1

                trace_entry = {
                    "sample_id": metrics["total_samples"],
                    "query": query,
                    "available_tools": [t.get("name") for t in tools_json] if isinstance(tools_json, list) else [],
                    "answers_str_raw": answers_str,
                    "ground_truth_parsing_error": ground_truth_parsing_error,
                    "parsed_answers": answers_json,
                    "raw_response": response,
                    "validator_output": parsed,
                    "outcome": outcome,
                    "error_category": error_category,
                    "latency_ms": latency_ms,
                    "generated_tokens": num_generated_tokens
                }
                
                f_trace.write(json.dumps(trace_entry) + "\n")
                f_trace.flush()
                
        # Generate final metrics report
        tps = metrics["total_generated_tokens"] / (metrics["total_inference_latency_ms"] / 1000) if metrics["total_inference_latency_ms"] > 0 else 0
        
        final_report = {
            "total_samples": metrics["total_samples"],
            "confusion_matrix": {
                "true_positives": metrics["true_positives"],
                "false_positives": metrics["false_positives"],
                "true_negatives": metrics["true_negatives"],
                "false_negatives": metrics["false_negatives"]
            },
            "error_distribution": {
                "syntax": metrics["error_syntax"],
                "missing_tool": metrics["error_missing_tool"],
                "hallucinated_tool": metrics["error_hallucinated_tool"],
                "hallucinated_param": metrics["error_hallucinated_param"],
                "missing_param": metrics["error_missing_param"],
                "type_mismatch": metrics["error_type_mismatch"],
                "value_mismatch": metrics["error_value_mismatch"]
            },
            "performance": {
                "avg_latency_ms": metrics["total_inference_latency_ms"] / max(1, metrics["total_samples"]),
                "tokens_per_second": tps
            }
        }
        
        report_path = os.path.join(reports_dir, "bfcl_advanced_metrics.json")
        with open(report_path, "w") as f:
            json.dump(final_report, f, indent=4)
            
        self._generate_plots(metrics, reports_dir)
        
        print(f"Evaluation complete. Advanced metrics saved to {report_path}")
        print(f"Debug trace saved to {trace_path}")
        return final_report
        
    def _generate_plots(self, metrics, reports_dir):
        try:
            # Confusion Matrix
            plt.figure(figsize=(6,5))
            matrix = [[metrics["true_positives"], metrics["false_positives"]],
                      [metrics["false_negatives"], metrics["true_negatives"]]]
            
            sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Positive", "Negative"],
                        yticklabels=["Positive", "Negative"])
            plt.ylabel('Actual (Ground Truth)')
            plt.xlabel('Predicted')
            plt.title('Agentic Confusion Matrix')
            plt.tight_layout()
            plt.savefig(os.path.join(reports_dir, "confusion_matrix.png"))
            plt.close()
            
            # Error Breakdown
            plt.figure(figsize=(10,6))
            error_types = ['Syntax', 'Hallucinated Tool', 'Hallucinated Param', 'Missing Param', 'Type Mismatch', 'Value Mismatch']
            error_counts = [
                metrics["error_syntax"],
                metrics["error_hallucinated_tool"],
                metrics["error_hallucinated_param"],
                metrics["error_missing_param"],
                metrics["error_type_mismatch"],
                metrics["error_value_mismatch"]
            ]
            
            sns.barplot(x=error_counts, y=error_types, palette="Reds_r")
            plt.title('Failure Breakdown Analysis')
            plt.xlabel('Number of Samples')
            plt.tight_layout()
            plt.savefig(os.path.join(reports_dir, "failure_breakdown.png"))
            plt.close()
        except Exception as e:
            print(f"Warning: Failed to generate plots. Ensure matplotlib/seaborn are installed. {e}")
