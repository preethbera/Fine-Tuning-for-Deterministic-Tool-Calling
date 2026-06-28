# Deterministic Agentic Tool Router

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![Unsloth](https://img.shields.io/badge/Powered_by-Unsloth-FF69B4.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)
![License](https://img.shields.io/badge/license-MIT-green)

A production-grade, highly modular Large Language Model (LLM) alignment repository specifically designed for **deterministic agentic tool routing**. This repository provides an end-to-end pipeline for dataset processing, parameter-efficient fine-tuning (QLoRA), and robust evaluation of open-source models for highly accurate function calling capabilities.

## 🌟 Key Features

- **End-to-End Pipeline**: Modular pipelines for data processing, training, and evaluation.
- **Optimized QLoRA Fine-Tuning**: Leverages [Unsloth](https://unsloth.ai/) for 2x-5x faster training with minimal VRAM usage. It uses native `load_in_4bit=True` with NF4 quantization, Double Quantization, and paged optimizers (`paged_adamw_8bit`).
- **Flexible Data Parsers**: Supports multiple dataset formats out of the box (e.g., ShareGPT, xLAM format).
- **Robust Evaluation**: Comprehensive evaluation pipeline that measures true/false positives, syntax errors, hallucinated parameters, type mismatches, and generation latency.
- **Configuration Driven**: Easily manage experiments and hyperparameters using YAML configurations.
- **Built-in Smoke Tests**: Validate dataset compatibility and model parsers before initiating long training runs.

## 🏗️ Repository Structure

```text
.
├── configs/                  # YAML configuration files
│   ├── base.yaml             # Main training & evaluation config
│   └── smoke_test.yaml       # Fast smoke test override config
├── src/
│   ├── adapters/             # Dataset parsers (xLAM, ShareGPT)
│   ├── core/                 # Configuration schemas, error handling, diagnostics
│   ├── models/               # RouterAgent (Unsloth wrapper) and Validator
│   └── pipelines/            # Core logic (Data, Training, Evaluation pipelines)
├── main.py                   # Central CLI entry point
├── pyproject.toml            # Project metadata and dependencies
└── requirements.txt          # Python requirements
```

## 🚀 Getting Started

### 1. Prerequisites

This repository requires an NVIDIA GPU (Ampere architecture or newer is recommended for optimal bfloat16 and Unsloth performance).

### 2. Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/your-org/deterministic-tool-router.git
cd deterministic-tool-router
pip install -r requirements.txt
# Or using pyproject.toml:
# pip install -e .
```

*Note: Unsloth installation can be environment-specific. Refer to the [official Unsloth installation guide](https://github.com/unslothai/unsloth) if you encounter issues.*

### 3. Configuration

Edit `configs/base.yaml` to set your desired model, dataset, and training hyperparameters:

```yaml
model:
  name_or_path: "unsloth/Qwen2.5-Coder-1.5B-Instruct-bnb-4bit"
  max_seq_length: 2048
  load_in_4bit: true # Enables QLoRA

data:
  dataset_name: "Beryex/xlam-function-calling-60k-sharegpt"
  
training:
  per_device_train_batch_size: 2
  learning_rate: 2.0e-4
  num_train_epochs: 1
  optim: "paged_adamw_8bit"
  # ...
```

## 🛠️ Usage

The central entry point is `main.py`, which manages the execution phases.

### Smoke Testing
Always run a smoke test before a full training job to ensure your data parsers and configs are correct. This will only run a few steps and evaluate a handful of samples.

```bash
python main.py --phase all --profile smoke_test
```

### Full Pipeline Execution

Run the complete pipeline (Process Data -> Train -> Evaluate):

```bash
python main.py --phase all --profile base
```

### Running Individual Phases

You can execute specific phases of the pipeline sequentially:

1. **Process Data**: Downloads, parses, formats, and tokenizes the dataset.
   ```bash
   python main.py --phase process
   ```
2. **Train Model**: Initiates QLoRA fine-tuning using `TRL`'s `SFTTrainer`.
   ```bash
   python main.py --phase train
   ```
3. **Evaluate Model**: Runs inference on the test set and calculates advanced function-calling metrics.
   ```bash
   python main.py --phase evaluate
   ```

## 📊 Evaluation & Metrics

The evaluation phase produces detailed metrics to ensure the determinism of your tool router. Reports are saved in `outputs/evaluation_reports/`.

The pipeline evaluates:
- **Accuracy**: True Positives, False Positives, False Negatives, True Negatives.
- **Failure Breakdown**: Categorizes errors into Syntax Errors, Hallucinated Tools/Params, Missing Params, Type Mismatches, and Value Mismatches.
- **Performance**: Measures Inference Latency (ms) and Tokens Per Second (TPS).

Visualizations (Confusion Matrix and Error Breakdown charts) are automatically generated for easy analysis.

## 🧠 QLoRA Implementation Details

This repository implements a textbook QLoRA setup following standard best practices:
- **NF4 Quantization**: Handled dynamically by Unsloth when `load_in_4bit=True`.
- **Double Quantization**: Enabled automatically by Unsloth to reduce VRAM overhead of quantization constants.
- **Paged Optimizers**: Explicitly configured using `paged_adamw_8bit` to offload optimizer states to CPU RAM during GPU memory spikes, preventing OOM errors.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
