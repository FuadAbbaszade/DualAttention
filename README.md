# Dual-Attention Whisper for Noise-Robust Speech Recognition

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/transformers-4.30+-orange.svg)](https://huggingface.co/transformers/)

## Overview

This project implements a **dual-attention mechanism** integrated into OpenAI's Whisper model to improve robustness against background noise and overlapping speech, particularly for low-resource languages like Azerbaijani.

**Key Features:**
- 🎯 Dual cross-attention mechanism for noise-robust ASR
- 🤗 Direct Hugging Face dataset integration
- ⚡ Optimized training pipeline for GPUs
- 🌍 Multi-language support (tested on Azerbaijani)
- 📊 Built-in WER/CER evaluation metrics

## Architecture

### Key Innovation

The standard Whisper decoder uses a single cross-attention mechanism. We introduce **two parallel cross-attention branches**:

1. **Primary Attention**: Focuses on linguistic alignment (clean speech features)
2. **Secondary Attention**: Focuses on noise-specific regions

Both attention heads process the same encoder output but learn different attention patterns:
- Primary attention learns to attend to speech-relevant features
- Secondary attention learns to attend to noise regions
- The model explicitly separates speech from noise before decoding

### Benefits

- ✅ Enhanced transcription accuracy in noisy conditions
- ✅ Preserves Whisper's alignment efficiency and generalization
- ✅ No changes to encoder (maintains pre-trained weights)
- ✅ Backward compatible with standard Whisper checkpoints

## Project Structure

```
Dual Attention/
├── README.md
├── requirements.txt
├── setup.py
├── src/
│   ├── __init__.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── dual_attention_decoder.py    # Modified decoder with dual attention
│   │   ├── dual_whisper.py              # Complete model wrapper
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py                   # Custom dataset for noisy audio
│   │   ├── collator.py                  # Data collator
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py                   # Custom trainer
│   │   ├── metrics.py                   # WER, CER metrics
├── scripts/
│   ├── train.py                         # Main training script
│   ├── inference.py                     # Inference script
│   ├── run_evaluation.py                # Evaluation script
│   ├── prepare_data.py                  # Data preparation (local & HuggingFace)
├── configs/
│   └── training_config.yaml             # Training configuration
└── notebooks/
    └── demo.ipynb                       # Interactive demo
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended: 16GB+ VRAM)
- PyTorch 2.0+

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd DualAttention

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Verify installation
python -c "from src.model.dual_whisper import DualAttentionWhisperForConditionalGeneration; print('✅ Installation successful!')"
```

## Quick Start

### Option 1: Using Hugging Face Datasets (Recommended)

```bash
# 1. Prepare data from Hugging Face
python scripts/prepare_data.py \
    --hf_dataset LocalDoc/azerbaijani_asr \
    --hf_split train \
    --hf_audio_column audio \
    --hf_text_column text \
    --language az \
    --output_dir ./data/azerbaijani_asr

# 2. Train the model
python scripts/train.py \
    --train_data data/azerbaijani_asr/train.json \
    --eval_data data/azerbaijani_asr/eval.json \
    --language az \
    --model_name openai/whisper-small \
    --output_dir outputs/azerbaijani_asr \
    --per_device_train_batch_size 16 \
    --num_train_epochs 3

# 3. Run inference
python scripts/inference.py \
    --model_path outputs/azerbaijani_asr \
    --audio_path test_audio.wav \
    --language az
```

### Option 2: Using Local Audio Files

```bash
# 1. Prepare local data
python scripts/prepare_data.py \
    --audio_dir /path/to/audio \
    --transcripts /path/to/transcripts.json \
    --output_dir ./data/processed \
    --language az

# 2. Train (same as above)
python scripts/train.py \
    --train_data data/processed/train.json \
    --eval_data data/processed/eval.json \
    --language az \
    --model_name openai/whisper-small \
    --output_dir outputs/model \
    --per_device_train_batch_size 16 \
    --num_train_epochs 3
```

## Training Configuration

### GPU Memory Requirements

| Model Size | Batch Size | GPU Memory | Training Speed |
|------------|------------|------------|----------------|
| **whisper-small** (244M) | 16 | ~12 GB | ~8 steps/sec |
| **whisper-small** (244M) | 24 | ~16 GB | ~10 steps/sec |
| **whisper-medium** (769M) | 8 | ~20 GB | ~4 steps/sec |
| **whisper-medium** (769M) | 16 | ~32 GB | ~5 steps/sec |
| **whisper-large** (1.5B) | 4 | ~24 GB | ~2 steps/sec |

*Tested on NVIDIA A100 40GB with FP16 training*

### Recommended Settings

**For A100 40GB / V100 32GB:**
```bash
# Whisper-small (fastest, good quality)
python scripts/train.py \
    --model_name openai/whisper-small \
    --per_device_train_batch_size 16 \
    --num_train_epochs 3

# Whisper-medium (best quality)
python scripts/train.py \
    --model_name openai/whisper-medium \
    --per_device_train_batch_size 12 \
    --num_train_epochs 3
```

**For RTX 3090 / RTX 4090 (24GB):**
```bash
python scripts/train.py \
    --model_name openai/whisper-small \
    --per_device_train_batch_size 12 \
    --num_train_epochs 3
```

### Optimization Features

- **Precision**: FP16 + TF32 automatic mixed precision
- **Optimizer**: AdamW Fused (fastest PyTorch optimizer)
- **Data Loading**: Multi-worker persistent data loading
- **Gradient Checkpointing**: Optional for larger models
- **Multi-GPU**: Automatic DDP support

## Model Architecture Details

### Standard Whisper Decoder Layer
```
Input → Self-Attention → Cross-Attention → FFN → Output
```

### Dual-Attention Decoder Layer
```
Input → Self-Attention → Primary Cross-Attn (speech) ──┐
                       → Secondary Cross-Attn (noise) ──┼→ Fusion → FFN → Output
                                                         │
                                              Gating Mechanism
```

The fusion mechanism learns to weight the two attention outputs dynamically.

## Citation

If you use this work, please cite:

```bibtex
@article{dual_attention_whisper,
  title={Dual-Attention Mechanism for Noise-Robust Whisper-Based Speech Recognition},
  year={2025}
}
```

## License

MIT License
