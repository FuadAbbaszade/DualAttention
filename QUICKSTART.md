# Quick Start Guide

## Installation

```bash
cd "Dual Attention"
pip install -r requirements.txt
pip install -e .
```

## Data Preparation

### Option 1: Use Hugging Face Dataset (Easiest)

The fastest way to get started:

```bash
python scripts/prepare_data.py \
    --hf_dataset LocalDoc/azerbaijani_asr \
    --hf_split train \
    --hf_audio_column audio \
    --hf_text_column text \
    --language az \
    --output_dir ./data/azerbaijani_asr
```

The dataset will be downloaded, cached, and automatically split into train/eval sets.

### Option 2: Prepare your own data

1. Organize your audio files in a directory:
```
/path/to/audio/
├── file1.wav
├── file2.wav
└── file3.wav
```

2. Create a transcripts file (JSON format):
```json
{
  "file1.wav": "first transcription",
  "file2.wav": "second transcription",
  "file3.wav": "third transcription"
}
```

3. Run data preparation:
```bash
python scripts/prepare_data.py \
    --audio_dir /path/to/audio \
    --transcripts /path/to/transcripts.json \
    --output_dir ./data/processed \
    --language az
```

### Option 3: Use existing dataset format

Create JSON files directly:

**train.json:**
```json
[
  {
    "audio_path": "/path/to/audio1.wav",
    "text": "transcription text",
    "language": "az",
    "duration": 5.2
  },
  ...
]
```

## Training

### Basic Training

```bash
python scripts/train.py \
    --model_name openai/whisper-small \
    --train_data ./data/processed/train.json \
    --eval_data ./data/processed/eval.json \
    --output_dir ./outputs \
    --language az \
    --max_steps 10000
```

### Advanced Training with Custom Settings

```bash
python scripts/train.py \
    --model_name openai/whisper-medium \
    --train_data ./data/processed/train.json \
    --eval_data ./data/processed/eval.json \
    --output_dir ./outputs/experiment1 \
    --language az \
    --freeze_encoder \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 32 \
    --learning_rate 5e-6 \
    --warmup_steps 1000 \
    --max_steps 20000 \
    --eval_steps 500 \
    --save_steps 500
```

### Multi-GPU Training

The training script automatically detects multiple GPUs and uses DataParallel:

```bash
# Will use all available GPUs
CUDA_VISIBLE_DEVICES=0,1 python scripts/train.py \
    --model_name openai/whisper-small \
    --train_data ./data/processed/train.json \
    --eval_data ./data/processed/eval.json \
    --output_dir ./outputs
```

### Monitor Training with TensorBoard

```bash
tensorboard --logdir ./outputs
```

Open http://localhost:6006 in your browser.

## Inference

### Single Audio File

```bash
python scripts/inference.py \
    --model_path ./outputs/checkpoint-10000 \
    --audio_path /path/to/audio.wav \
    --language az
```

### Batch Inference on Directory

```bash
python scripts/inference.py \
    --model_path ./outputs/checkpoint-10000 \
    --audio_path /path/to/audio_dir \
    --language az \
    --output_file results.txt
```

### With Custom Generation Settings

```bash
python scripts/inference.py \
    --model_path ./outputs/checkpoint-10000 \
    --audio_path /path/to/audio.wav \
    --language az \
    --num_beams 10 \
    --task transcribe
```

## Evaluation

```bash
python scripts/run_evaluation.py \
    --model_path ./outputs/checkpoint-10000 \
    --test_data ./data/processed/eval.json \
    --language az \
    --output_file evaluation_results.txt
```

**Note:** The script was renamed to `run_evaluation.py` to avoid naming conflicts.

## Python API Usage

### Training

```python
from transformers import WhisperProcessor, Seq2SeqTrainingArguments, Seq2SeqTrainer
from src.model.dual_whisper import create_dual_attention_whisper
from src.data.dataset import NoisyAudioDataset
from src.data.collator import DataCollatorSpeechSeq2SeqWithPadding
from src.training.metrics import compute_metrics

# Load model
model = create_dual_attention_whisper(
    model_name="openai/whisper-small",
    freeze_encoder=True
)

# Load processor
processor = WhisperProcessor.from_pretrained("openai/whisper-small")

# Load datasets
train_dataset = NoisyAudioDataset(
    data_path="./data/processed/train.json",
    processor=processor,
    language="az"
)

# ... continue with training setup
```

### Inference

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
    audio, 
    sampling_rate=16000, 
    return_tensors="pt"
).input_features

# Generate
with torch.no_grad():
    generated_ids = model.generate(
        input_features,
        language="az",
        task="transcribe"
    )

# Decode
transcription = processor.batch_decode(
    generated_ids, 
    skip_special_tokens=True
)[0]

print(transcription)
```

## Troubleshooting

### Out of Memory (OOM)

1. Reduce batch size:
```bash
--per_device_train_batch_size 8
```

2. Enable gradient checkpointing (edit `train.py`):
```python
gradient_checkpointing=True
```

3. Use smaller model:
```bash
--model_name openai/whisper-tiny
```

### Slow Training

1. Use fewer dataloader workers if CPU is bottleneck
2. Reduce evaluation frequency: `--eval_steps 2000`
3. Use `--generation_num_beams 1` for faster eval
4. Enable TF32 (automatic on Ampere GPUs)

### Low Accuracy

1. Train longer: increase `--max_steps`
2. Use larger model: `whisper-medium` or `whisper-large`
3. Collect more training data
4. Unfreeze encoder after initial training
5. Reduce learning rate: `--learning_rate 1e-6`

## GPU Requirements

**Quick reference for model selection:**

| GPU | Recommended Model | Batch Size |
|-----|------------------|------------|
| RTX 3060 12GB | whisper-tiny | 16-24 |
| RTX 3090 24GB | whisper-small | 12-16 |
| A100 40GB | whisper-small | 16-24 |
| A100 40GB | whisper-medium | 12 |
| A100 80GB | whisper-large | 8 |

## Tips for Best Results

1. **Freeze encoder initially**: Let the dual-attention decoder adapt first
2. **Choose right model size**: Start with whisper-small for A100 40GB
3. **Use appropriate batch size**: See GPU requirements table above
4. **Train with epochs**: Use `--num_train_epochs 3` for complete training
5. **Fine-tune in stages**:
   - Stage 1: Freeze encoder, train decoder (5k steps)
   - Stage 2: Unfreeze encoder, full fine-tuning (5k steps)
6. **Monitor overfitting**: Watch eval WER, use early stopping
7. **Language-specific tokens**: Always set `--language` correctly

## Next Steps

- Check out `notebooks/demo.ipynb` for interactive examples
- Read the paper: `Dual_Attention_Whisper_Research.docx`
- Experiment with different freezing strategies
- Try different base models (tiny, small, medium, large)
