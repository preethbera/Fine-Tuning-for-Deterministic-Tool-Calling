import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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
    
    parser_impl = SharegptParser() if "sharegpt" in config.data.dataset_name.lower() else XlamParser()
    
    if args.phase in ['process', 'all']:
        print("\n=== Phase: Process ===")
        agent = RouterAgent(config)
        _, tokenizer = agent.initialize_model()
        
        data_pipeline = DataPipeline(config, parser_impl, tokenizer)
        data_pipeline.process()
        enforce_gc()
        
    if args.phase in ['train', 'all']:
        print("\n=== Phase: Train ===")
        training_pipeline = TrainingPipeline(config)
        training_pipeline.execute()
        enforce_gc()
        
    if args.phase in ['evaluate', 'all']:
        print("\n=== Phase: Evaluate ===")
        evaluation_pipeline = EvaluationPipeline(config, parser_impl)
        evaluation_pipeline.execute(num_samples=100 if args.profile == "base" else 10)
        enforce_gc()

if __name__ == "__main__":
    main()
