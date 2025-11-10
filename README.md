# Dual-Attention Whisper for Noise-Robust Speech Recognition

## Overview

This project implements a **dual-attention mechanism** integrated into OpenAI's Whisper model to improve robustness against background noise and overlapping speech, particularly for low-resource languages like Azerbaijani.

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
│   ├── evaluate.py                      # Evaluation script
│   ├── prepare_data.py                  # Data preparation
├── configs/
│   └── training_config.yaml             # Training configuration
└── notebooks/
    └── demo.ipynb                       # Interactive demo
```

## Installation

```bash
# Clone and install
cd "Dual Attention"
pip install -r requirements.txt
pip install -e .
```

## Quick Start

### 1. Prepare Your Data

```bash
python scripts/prepare_data.py \
    --audio_dir /path/to/audio \
    --transcripts /path/to/transcripts.json \
    --output_dir ./data/processed
```

### 2. Train the Model

```bash
python scripts/train.py \
    --model_name openai/whisper-small \
    --data_dir ./data/processed \
    --output_dir ./outputs \
    --num_gpus 2
```

### 3. Run Inference

```bash
python scripts/inference.py \
    --model_path ./outputs/checkpoint-10000 \
    --audio_path /path/to/test.wav \
    --language az
```

## Training Configuration

The training script uses optimized settings for maximum GPU utilization:

- **Batch Size**: 16 per device
- **Gradient Accumulation**: 1 step
- **Learning Rate**: 5e-6 with warmup
- **Optimization**: AdamW Fused (fastest)
- **Precision**: FP16 + TF32
- **Multi-GPU**: DDP with optimized settings
- **Data Loading**: 16 workers with persistent workers

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
