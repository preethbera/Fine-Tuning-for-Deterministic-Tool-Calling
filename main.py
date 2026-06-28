import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import unsloth
import argparse
import gc
import torch
from src.core.config import load_config
from src.adapters.xlam_parser import XlamParser
from src.adapters.sharegpt_parser import SharegptParser
from src.pipelines.data_pipeline import DataPipeline
from src.pipelines.training_pipeline import TrainingPipeline
from src.pipelines.evaluation_pipeline import EvaluationPipeline
from src.models.router_agent import RouterAgent

def enforce_gc():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser(description="Enterprise Deterministic Agentic Tool Router Pipeline")
    parser.add_argument('--phase', type=str, choices=['process', 'train', 'evaluate', 'all'], default='all',
                        help='Which phase of the pipeline to run')
    parser.add_argument('--profile', type=str, choices=['base', 'smoke_test'], default='base',
                        help='Which configuration profile to use')
    
    args = parser.parse_args()
    
    config = load_config(args.profile)
    is_smoke_test = (args.profile == "smoke_test")
    
    parser_impl = SharegptParser() if "sharegpt" in config.data.dataset_name.lower() else XlamParser()
    
    # Initialize diagnostic report for smoke tests
    diag = None
    if is_smoke_test:
        from src.core.diagnostic import DiagnosticReport
        diag = DiagnosticReport(config, parser_impl)
        diag.capture_config()
    
    if args.phase in ['process', 'all']:
        print("\n=== Phase: Process ===")
        agent = RouterAgent(config)
        _, tokenizer = agent.initialize_model()
        
        from datasets import load_dataset
        raw_dataset = load_dataset(config.data.dataset_name, split="train")
        if config.data.dataset_limit:
            raw_dataset = raw_dataset.select(range(min(config.data.dataset_limit, len(raw_dataset))))
        
        if diag:
            diag.analyze_dataset(raw_dataset)
            adapter_ok = diag.validate_adapter(raw_dataset)
            
            data_pipeline = DataPipeline(config, parser_impl, tokenizer)
            diag.validate_formatted_prompts(data_pipeline, raw_dataset)
            
            if not adapter_ok:
                print("\n[SMOKE TEST ABORT] Adapter validation FAILED. Fix the parser before training.")
                diag.save()
                return
        
        data_pipeline = DataPipeline(config, parser_impl, tokenizer)
        data_pipeline.process()
        enforce_gc()
        
    if args.phase in ['train', 'all']:
        print("\n=== Phase: Train ===")
        training_pipeline = TrainingPipeline(config)
        train_result = training_pipeline.execute()
        
        if diag:
            diag.capture_training_results(train_result)
        
        enforce_gc()
        
    if args.phase in ['evaluate', 'all']:
        print("\n=== Phase: Evaluate ===")
        evaluation_pipeline = EvaluationPipeline(config, parser_impl)
        eval_report = evaluation_pipeline.execute(num_samples=100 if not is_smoke_test else 10)
        
        if diag:
            diag.capture_evaluation_results(eval_report)
        
        enforce_gc()
    
    if diag:
        diag.save()

if __name__ == "__main__":
    main()
