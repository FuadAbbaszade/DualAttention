# Complete Usage Guide for Dual-Attention Whisper

## 📋 Table of Contents

1. [Installation](#installation)
2. [Data Preparation](#data-preparation)
3. [Training](#training)
4. [Inference](#inference)
5. [Evaluation](#evaluation)
6. [Advanced Topics](#advanced-topics)
7. [Troubleshooting](#troubleshooting)

---

## 🔧 Installation

### Step 1: Install Dependencies

```bash
cd "Dual Attention"
pip install -r requirements.txt
pip install -e .
```

### Step 2: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "from src.model.dual_whisper import DualAttentionWhisperForConditionalGeneration; print('✅ Dual-Attention Whisper imported successfully')"
```

---

## 📊 Data Preparation

### Format Your Data

Your data should be in JSON format with the following structure:

```json
[
  {
    "audio_path": "/absolute/path/to/audio1.wav",
    "text": "transcription text in Azerbaijani",
    "language": "az",
    "duration": 5.2
  }
]
```

### Option 1: Auto-Prepare from Audio Files

If you have audio files and a separate transcript file:

```bash
python scripts/prepare_data.py \
    --audio_dir /path/to/your/audio/files \
    --transcripts /path/to/transcripts.json \
    --output_dir ./data/processed \
    --language az \
    --min_duration 0.5 \
    --max_duration 30.0 \
    --train_split 0.9
```

**Transcript formats supported:**

1. JSON format:
```json
{
  "audio1.wav": "first transcription",
  "audio2.wav": "second transcription"
}
```

2. TSV format (text file):
```
audio1.wav	first transcription
audio2.wav	second transcription
```

### Option 2: Manual Preparation

Create `train.json` and `eval.json` files manually:

```bash
mkdir -p data/processed
# Edit data/processed/train.json
# Edit data/processed/eval.json
```

---

## 🏋️ Training

### Basic Training (Recommended Start)

```bash
python scripts/train.py \
    --model_name openai/whisper-small \
    --train_data ./data/processed/train.json \
    --eval_data ./data/processed/eval.json \
    --output_dir ./outputs/experiment1 \
    --language az \
    --freeze_encoder \
    --max_steps 10000 \
    --per_device_train_batch_size 16 \
    --learning_rate 5e-6
```

### Training with Your Configuration (From Your Example)

```bash
python scripts/train.py \
    --model_name openai/whisper-small \
    --train_data ./data/processed/train.json \
    --eval_data ./data/processed/eval.json \
    --output_dir ./outputs \
    --language az \
    --freeze_encoder \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 32 \
    --learning_rate 5e-6 \
    --warmup_steps 1000 \
    --max_steps 10000 \
    --eval_steps 1000 \
    --save_steps 1000 \
    --logging_steps 10
```

### Multi-GPU Training

```bash
# Use all GPUs
python scripts/train.py \
    --model_name openai/whisper-small \
    --train_data ./data/processed/train.json \
    --eval_data ./data/processed/eval.json \
    --output_dir ./outputs

# Use specific GPUs
CUDA_VISIBLE_DEVICES=0,1 python scripts/train.py \
    --model_name openai/whisper-small \
    --train_data ./data/processed/train.json \
    --eval_data ./data/processed/eval.json \
    --output_dir ./outputs
```

### Monitor Training with TensorBoard

```bash
# Open a new terminal
tensorboard --logdir ./outputs

# Open browser to http://localhost:6006
```

### Two-Stage Training (Best Results)

**Stage 1: Train dual-attention decoder only**
```bash
python scripts/train.py \
    --model_name openai/whisper-small \
    --train_data ./data/processed/train.json \
    --eval_data ./data/processed/eval.json \
    --output_dir ./outputs/stage1 \
    --language az \
    --freeze_encoder \
    --max_steps 5000 \
    --learning_rate 5e-6
```

**Stage 2: Fine-tune entire model**
```bash
# Manually edit train.py to load from stage1 checkpoint
# Set freeze_encoder=False
python scripts/train.py \
    --model_name ./outputs/stage1/checkpoint-5000 \
    --train_data ./data/processed/train.json \
    --eval_data ./data/processed/eval.json \
    --output_dir ./outputs/stage2 \
    --language az \
    --max_steps 5000 \
    --learning_rate 1e-6
```

---

## 🎤 Inference

### Single Audio File

```bash
python scripts/inference.py \
    --model_path ./outputs/checkpoint-10000 \
    --audio_path /path/to/audio.wav \
    --language az \
    --num_beams 5
```

### Batch Inference on Directory

```bash
python scripts/inference.py \
    --model_path ./outputs/checkpoint-10000 \
    --audio_path /path/to/audio_directory \
    --language az \
    --num_beams 5 \
    --output_file transcriptions.txt
```

### Python API

```python
import torch
from transformers import WhisperProcessor
from src.model.dual_whisper import DualAttentionWhisperForConditionalGeneration
import librosa

# Load model
processor = WhisperProcessor.from_pretrained("./outputs/checkpoint-10000")
model = DualAttentionWhisperForConditionalGeneration.from_pretrained(
    "./outputs/checkpoint-10000"
)
model.eval()

# Load audio
audio, sr = librosa.load("audio.wav", sr=16000)

# Extract features
input_features = processor.feature_extractor(
    audio, sampling_rate=16000, return_tensors="pt"
).input_features

# Generate
with torch.no_grad():
    generated_ids = model.generate(
        input_features,
        language="az",
        task="transcribe",
        num_beams=5
    )

# Decode
transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(transcription)
```

---

## 📊 Evaluation

### Evaluate on Test Set

```bash
python scripts/evaluate.py \
    --model_path ./outputs/checkpoint-10000 \
    --test_data ./data/processed/eval.json \
    --language az \
    --output_file evaluation_results.txt
```

This will output:
- WER (Word Error Rate)
- CER (Character Error Rate)
- Sample predictions vs references

---

## 🔬 Advanced Topics

### 1. Custom Model Configuration

Edit `src/model/dual_whisper.py` to modify:
- Gating mechanism architecture
- Number of attention heads
- Hidden dimensions

### 2. Data Augmentation

Enable noise injection during training by modifying the collator:

```python
# In train.py, replace:
from data.collator import DataCollatorSpeechSeq2SeqWithPadding

# With:
from data.collator import DataCollatorSpeechSeq2SeqWithPaddingAndNoise

data_collator = DataCollatorSpeechSeq2SeqWithPaddingAndNoise(
    processor=processor,
    add_noise=True,
    noise_prob=0.5,
    noise_level=0.01
)
```

### 3. Different Base Models

Try different Whisper model sizes:

```bash
# Tiny (39M params) - fastest, lowest accuracy
--model_name openai/whisper-tiny

# Small (244M params) - good balance
--model_name openai/whisper-small

# Medium (769M params) - better accuracy
--model_name openai/whisper-medium

# Large (1550M params) - best accuracy, slowest
--model_name openai/whisper-large-v2
```

### 4. Hyperparameter Tuning

Key hyperparameters to tune:

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| learning_rate | 5e-6 | 1e-6 to 1e-5 | Higher = faster but less stable |
| batch_size | 16 | 4 to 32 | Higher = faster, needs more VRAM |
| warmup_steps | 1000 | 100 to 2000 | More warmup = more stable |
| num_beams | 5 | 1 to 10 | Higher = better quality, slower |

### 5. Memory Optimization

If you run out of GPU memory:

```bash
# Reduce batch size
--per_device_train_batch_size 8

# Use gradient accumulation
--gradient_accumulation_steps 2

# Enable gradient checkpointing (edit train.py)
gradient_checkpointing=True

# Use smaller model
--model_name openai/whisper-tiny
```

---

## 🐛 Troubleshooting

### Problem: Out of Memory (OOM)

**Solutions:**
1. Reduce batch size: `--per_device_train_batch_size 8`
2. Use gradient accumulation: `--gradient_accumulation_steps 2`
3. Use smaller model: `--model_name openai/whisper-tiny`
4. Reduce audio length: Edit `max_audio_length` in dataset

### Problem: Training is Very Slow

**Solutions:**
1. Reduce number of workers: `--dataloader_num_workers 8`
2. Disable persistent workers (edit `train.py`)
3. Use faster evaluation: `--generation_num_beams 1`
4. Reduce eval frequency: `--eval_steps 2000`
5. Check if FP16 is enabled (it should be automatic)

### Problem: Poor Accuracy

**Solutions:**
1. Train longer: `--max_steps 20000`
2. Use larger model: `--model_name openai/whisper-medium`
3. Check data quality (transcripts should be accurate)
4. Try lower learning rate: `--learning_rate 1e-6`
5. Unfreeze encoder after initial training
6. Add more training data

### Problem: Model Not Learning

**Check:**
1. Loss is decreasing? Check TensorBoard
2. Data is loaded correctly? Check dataset size
3. Learning rate too high/low? Try 1e-6 to 1e-5
4. Gradients flowing? Check for frozen parameters

### Problem: Import Errors

```bash
# Reinstall in development mode
pip install -e .

# Check Python path
python -c "import sys; print(sys.path)"
```

### Problem: CUDA Errors

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Check CUDA version
nvidia-smi

# Reinstall PyTorch with correct CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 📝 Example Workflow

### Complete Training Pipeline

```bash
# 1. Prepare data
python scripts/prepare_data.py \
    --audio_dir /path/to/audio \
    --transcripts /path/to/transcripts.json \
    --output_dir ./data/processed \
    --language az

# 2. Train model (stage 1)
python scripts/train.py \
    --model_name openai/whisper-small \
    --train_data ./data/processed/train.json \
    --eval_data ./data/processed/eval.json \
    --output_dir ./outputs \
    --language az \
    --freeze_encoder \
    --max_steps 10000

# 3. Evaluate
python scripts/evaluate.py \
    --model_path ./outputs/checkpoint-10000 \
    --test_data ./data/processed/eval.json \
    --language az

# 4. Run inference
python scripts/inference.py \
    --model_path ./outputs/checkpoint-10000 \
    --audio_path /path/to/test_audio.wav \
    --language az
```

---

## 🎯 Quick Reference

### Training Commands

```bash
# Basic training
python scripts/train.py --train_data DATA --eval_data DATA --output_dir OUT

# Resume from checkpoint
python scripts/train.py --model_name ./outputs/checkpoint-5000 --train_data DATA --eval_data DATA

# Multi-GPU
CUDA_VISIBLE_DEVICES=0,1 python scripts/train.py --train_data DATA --eval_data DATA
```

### Inference Commands

```bash
# Single file
python scripts/inference.py --model_path MODEL --audio_path AUDIO

# Batch processing
python scripts/inference.py --model_path MODEL --audio_path DIR --output_file OUT
```

### Evaluation Commands

```bash
# Standard evaluation
python scripts/evaluate.py --model_path MODEL --test_data DATA

# With output file
python scripts/evaluate.py --model_path MODEL --test_data DATA --output_file results.txt
```

---

## 📚 Additional Resources

- **README.md**: Project overview
- **QUICKSTART.md**: Getting started guide
- **PROJECT_SUMMARY.md**: Detailed architecture and theory
- **notebooks/demo.ipynb**: Interactive examples
- **configs/training_config.yaml**: Configuration reference

---

**Happy Training! 🚀**
