import os
import json
from datetime import datetime
from src.core.config import AppConfig
from src.adapters.base_parser import BaseDatasetParser

class DiagnosticReport:
    """Generates a structured diagnostic report during smoke test runs.
    Captures dataset schema, sample examples, training config, and training/eval results
    to catch data pipeline bugs BEFORE committing to a full training run.
    """
    
    def __init__(self, config: AppConfig, parser: BaseDatasetParser, report_dir: str = "outputs/smoke_test_report"):
        self.config = config
        self.parser = parser
        self.report_dir = report_dir
        self.report = {
            "generated_at": datetime.now().isoformat(),
            "profile": "smoke_test",
            "config_snapshot": {},
            "dataset_analysis": {},
            "data_pipeline_validation": {},
            "training_summary": {},
            "evaluation_summary": {}
        }
        os.makedirs(self.report_dir, exist_ok=True)
        
    def capture_config(self):
        """Snapshot the full resolved configuration."""
        self.report["config_snapshot"] = {
            "model": self.config.model.model_dump(),
            "lora": self.config.lora.model_dump(),
            "data": self.config.data.model_dump(),
            "training": self.config.training.model_dump()
        }
    
    def analyze_dataset(self, dataset, num_examples: int = 3):
        """Analyze raw dataset schema and capture sample records."""
        raw_columns = list(dataset.column_names) if hasattr(dataset, 'column_names') else []
        first_row = dataset[0] if len(dataset) > 0 else {}
        
        column_types = {}
        for col in raw_columns:
            val = first_row.get(col)
            column_types[col] = type(val).__name__
        
        self.report["dataset_analysis"]["source"] = self.config.data.dataset_name
        self.report["dataset_analysis"]["total_rows"] = len(dataset)
        self.report["dataset_analysis"]["columns"] = raw_columns
        self.report["dataset_analysis"]["column_types"] = column_types
        
        raw_examples = []
        for i in range(min(num_examples, len(dataset))):
            row = dataset[i]
            serializable_row = {}
            for k, v in row.items():
                if isinstance(v, (str, int, float, bool, type(None))):
                    serializable_row[k] = v
                elif isinstance(v, list):
                    serializable_row[k] = str(v)[:500]
                else:
                    serializable_row[k] = str(v)[:500]
            raw_examples.append(serializable_row)
        self.report["dataset_analysis"]["raw_examples"] = raw_examples
    
    def validate_adapter(self, dataset, num_samples: int = 5):
        """Run the adapter on N samples and log exactly what it produces.
        This is the critical check that catches schema mismatches.
        """
        results = []
        for i in range(min(num_samples, len(dataset))):
            row = dataset[i]
            entry = {"sample_index": i, "status": "unknown"}
            
            try:
                record = self.parser.transform(row)
                
                # Validate tool names are NOT null/unknown
                tool_names = [t.get("name") for t in record.tools if isinstance(t, dict)]
                has_null_tools = any(n is None or n == "unknown" for n in tool_names)
                
                # Validate answers are parseable
                answers_parsed = []
                answers_parse_error = None
                if record.answers and record.answers != "[]":
                    try:
                        answers_parsed = json.loads(record.answers)
                    except json.JSONDecodeError as e:
                        answers_parse_error = str(e)
                
                answer_tool_names = []
                if isinstance(answers_parsed, list):
                    for a in answers_parsed:
                        if isinstance(a, dict):
                            answer_tool_names.append(a.get("name"))
                
                entry["status"] = "OK" if not has_null_tools and not answers_parse_error else "WARNING"
                entry["query"] = record.query[:200]
                entry["tool_count"] = len(record.tools)
                entry["tool_names"] = tool_names
                entry["has_null_tools"] = has_null_tools
                entry["answers_raw"] = record.answers[:300] if record.answers else ""
                entry["answers_parsed_count"] = len(answers_parsed) if isinstance(answers_parsed, list) else 0
                entry["answer_tool_names"] = answer_tool_names
                entry["answers_parse_error"] = answers_parse_error
                
            except Exception as e:
                entry["status"] = "ERROR"
                entry["error"] = str(e)
            
            results.append(entry)
        
        # Summary flags
        all_ok = all(r["status"] == "OK" for r in results)
        null_tool_count = sum(1 for r in results if r.get("has_null_tools", False))
        parse_error_count = sum(1 for r in results if r.get("answers_parse_error"))
        
        self.report["data_pipeline_validation"] = {
            "samples_checked": len(results),
            "all_passed": all_ok,
            "null_tool_name_count": null_tool_count,
            "answer_parse_error_count": parse_error_count,
            "details": results
        }
        
        return all_ok

    def validate_formatted_prompts(self, data_pipeline, dataset, num_samples: int = 3):
        """Validate the final formatted training prompts to catch formatting bugs."""
        prompt_samples = []
        
        for i in range(min(num_samples, len(dataset))):
            row = dataset[i]
            entry = {"sample_index": i, "status": "unknown"}
            
            try:
                record = self.parser.transform(row)
                prompt = data_pipeline.generate_prompt_for_record(record)
                
                has_thought = "<thought>" in prompt and "</thought>" in prompt
                has_tool_call = "<tool_call>" in prompt and "</tool_call>" in prompt
                has_abort = "[ABORT:" in prompt
                has_unknown = '"unknown"' in prompt or ": unknown" in prompt.lower()
                
                is_positive = not (has_abort and not has_tool_call)
                
                entry["status"] = "OK"
                entry["is_positive_case"] = is_positive
                entry["has_thought_tags"] = has_thought
                entry["has_tool_call_tags"] = has_tool_call
                entry["has_abort"] = has_abort
                entry["has_unknown_token"] = has_unknown
                entry["prompt_length_chars"] = len(prompt)
                entry["prompt_preview"] = prompt[:800]
                
                if has_unknown:
                    entry["status"] = "WARNING"
                    entry["warning"] = "Prompt contains 'unknown' token — possible schema mismatch"
                    
            except Exception as e:
                entry["status"] = "ERROR"
                entry["error"] = str(e)
            
            prompt_samples.append(entry)
        
        self.report["data_pipeline_validation"]["formatted_prompt_samples"] = prompt_samples
    
    def capture_training_results(self, train_result):
        """Capture training metrics from the trainer result object."""
        if train_result is None:
            self.report["training_summary"] = {"status": "skipped_or_failed"}
            return
            
        metrics = {}
        if hasattr(train_result, 'metrics'):
            metrics = train_result.metrics
        elif isinstance(train_result, dict):
            metrics = train_result
            
        self.report["training_summary"] = {
            "status": "completed",
            "metrics": metrics
        }
    
    def capture_evaluation_results(self, eval_report):
        """Capture the full evaluation report."""
        self.report["evaluation_summary"] = eval_report if eval_report else {"status": "skipped_or_failed"}
    
    def save(self):
        """Save the final diagnostic report as a structured JSON file."""
        report_path = os.path.join(self.report_dir, "smoke_test_diagnostic.json")
        with open(report_path, "w") as f:
            json.dump(self.report, f, indent=4, default=str)
        
        print(f"\n{'='*60}")
        print("SMOKE TEST DIAGNOSTIC REPORT")
        print(f"{'='*60}")
        
        # Config
        print(f"\nModel: {self.config.model.name_or_path}")
        print(f"Dataset: {self.config.data.dataset_name}")
        print(f"Dataset Limit: {self.config.data.dataset_limit}")
        
        # Dataset
        ds = self.report.get("dataset_analysis", {})
        print(f"\nDataset Columns: {ds.get('columns', [])}")
        print(f"Column Types: {ds.get('column_types', {})}")
        
        # Adapter Validation
        dv = self.report.get("data_pipeline_validation", {})
        status = "PASS" if dv.get("all_passed") else "FAIL"
        print(f"\nAdapter Validation: {status}")
        print(f"  Null Tool Names: {dv.get('null_tool_name_count', 'N/A')}")
        print(f"  Answer Parse Errors: {dv.get('answer_parse_error_count', 'N/A')}")
        
        if not dv.get("all_passed"):
            print("\n  DETAILS:")
            for d in dv.get("details", []):
                if d.get("status") != "OK":
                    print(f"    Sample {d['sample_index']}: {d['status']}")
                    if d.get("has_null_tools"):
                        print(f"      Tool names: {d.get('tool_names')}")
                    if d.get("answers_parse_error"):
                        print(f"      Answer parse error: {d['answers_parse_error']}")
        
        # Formatted Prompts
        prompts = dv.get("formatted_prompt_samples", [])
        if prompts:
            print(f"\nFormatted Prompt Samples:")
            for p in prompts:
                marker = "WARN" if p.get("has_unknown_token") else "OK"
                case_type = "positive" if p.get("is_positive_case") else "negative"
                print(f"  [{marker}] Sample {p['sample_index']}: {case_type}, {p.get('prompt_length_chars', 0)} chars")
                if p.get("warning"):
                    print(f"       {p['warning']}")
        
        # Training
        ts = self.report.get("training_summary", {})
        if ts.get("status") == "completed":
            m = ts.get("metrics", {})
            print(f"\nTraining: COMPLETED")
            print(f"  Loss: {m.get('train_loss', 'N/A')}")
            print(f"  Runtime: {m.get('train_runtime', 'N/A')}s")
            print(f"  Samples/sec: {m.get('train_samples_per_second', 'N/A')}")
        
        # Evaluation
        ev = self.report.get("evaluation_summary", {})
        if ev and ev.get("total_samples"):
            cm = ev.get("confusion_matrix", {})
            print(f"\nEvaluation ({ev.get('total_samples', 0)} samples):")
            print(f"  True Positives:  {cm.get('true_positives', 0)}")
            print(f"  True Negatives:  {cm.get('true_negatives', 0)}")
            print(f"  False Positives: {cm.get('false_positives', 0)}")
            print(f"  False Negatives: {cm.get('false_negatives', 0)}")
        
        print(f"\nFull report saved to: {report_path}")
        print(f"{'='*60}\n")
        
        return report_path
