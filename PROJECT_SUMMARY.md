# Dual-Attention Whisper Project Summary

## 🎯 Project Goal

Implement a **dual-attention mechanism** in OpenAI's Whisper model to improve speech recognition accuracy in noisy environments, particularly for low-resource languages like Azerbaijani.

## 🔬 Research Motivation

Standard Whisper performs poorly in noisy or overlapping speech scenarios. The dual-attention approach explicitly models noise-specific characteristics separately from linguistic content, enabling better separation of speech from background noise.

## 🏗️ Architecture

### Standard Whisper Decoder
```
Input → Self-Attention → Cross-Attention → FFN → Output
                              ↑
                         Encoder Output
```

### Dual-Attention Decoder (Our Implementation)
```
Input → Self-Attention → Primary Cross-Attention (speech) ──┐
                       → Secondary Cross-Attention (noise) ──┼→ Gating → FFN → Output
                              ↑                    ↑         │
                              └────────────────────┘         │
                                 Encoder Output         Learned Fusion
```

### Key Components

1. **Primary Cross-Attention**: Attends to speech-relevant features
2. **Secondary Cross-Attention**: Attends to noise-specific regions
3. **Gating Mechanism**: Dynamically weighs primary vs secondary attention
4. **Encoder**: Unchanged (preserves pre-trained representations)

## 📁 Project Structure

```
Dual Attention/
├── README.md                    # Project overview
├── QUICKSTART.md               # Getting started guide
├── PROJECT_SUMMARY.md          # This file
├── requirements.txt            # Python dependencies
├── setup.py                    # Package installation
├── 
├── src/                        # Source code
│   ├── model/
│   │   ├── dual_attention_decoder.py    # Dual attention decoder layer
│   │   └── dual_whisper.py              # Complete model wrapper
│   ├── data/
│   │   ├── dataset.py                   # Dataset class
│   │   └── collator.py                  # Data collator
│   └── training/
│       └── metrics.py                   # WER/CER metrics
│
├── scripts/                    # Executable scripts
│   ├── train.py               # Training script
│   ├── inference.py           # Inference script
│   ├── evaluate.py            # Evaluation script
│   └── prepare_data.py        # Data preparation
│
├── configs/
│   └── training_config.yaml   # Training configuration
│
└── notebooks/
    └── demo.ipynb             # Interactive demo
```

## 🔑 Key Features

### 1. **Dual Cross-Attention**
- Two parallel attention heads in decoder
- One focuses on clean speech patterns
- Other focuses on noise characteristics
- Both attend to same encoder output but learn different patterns

### 2. **Gated Fusion**
- Learnable gating mechanism
- Dynamically weights primary vs secondary attention
- Formula: `output = α × primary + (1-α) × secondary`
- α learned from query and both attention outputs

### 3. **Backward Compatible**
- Loads pre-trained Whisper checkpoints
- Encoder unchanged (frozen during training)
- Primary attention initialized from pre-trained weights
- Only secondary attention and gates trained from scratch

### 4. **Optimized Training**
- FP16 + TF32 mixed precision
- Fused AdamW optimizer
- Multi-GPU support with DDP
- Persistent data workers
- Tensorboard logging

## 📊 Training Strategy

### Stage 1: Warm Start (Recommended)
```bash
# Freeze encoder, train decoder with dual attention
python scripts/train.py \
    --freeze_encoder \
    --max_steps 5000 \
    --learning_rate 5e-6
```

### Stage 2: Full Fine-tuning (Optional)
```bash
# Unfreeze encoder for full model fine-tuning
python scripts/train.py \
    --model_path ./outputs/checkpoint-5000 \
    --max_steps 5000 \
    --learning_rate 1e-6
```

## 🎯 Expected Results

Based on the research paper:
- **Standard Whisper on noisy data**: 25-35% WER
- **Dual-Attention Whisper**: 15-25% WER (40% relative improvement)
- **Clean data**: Similar performance to standard Whisper
- **Noisy data**: Significant improvement

## 💡 Implementation Highlights

### Dual Attention Layer
```python
class DualAttentionDecoderLayer:
    def forward(self, hidden_states, encoder_output):
        # Self-attention
        hidden = self.self_attn(hidden_states)
        
        # Dual cross-attention
        primary = self.cross_attn_primary(hidden, encoder_output)
        secondary = self.cross_attn_secondary(hidden, encoder_output)
        
        # Gated fusion
        fused = self.gate(hidden, primary, secondary)
        
        # FFN
        output = self.ffn(fused)
        return output
```

### Gating Mechanism
```python
class DualCrossAttentionGate:
    def forward(self, query, primary, secondary):
        # Concatenate features
        gate_input = concat([query, primary, secondary])
        
        # Compute gate (0 to 1)
        alpha = sigmoid(self.gate_network(gate_input))
        
        # Weighted fusion
        return alpha * primary + (1 - alpha) * secondary
```

## 🔧 Customization Options

### 1. Different Base Models
```python
# Try different Whisper sizes
model = create_dual_attention_whisper("openai/whisper-tiny")    # 39M params
model = create_dual_attention_whisper("openai/whisper-small")   # 244M params
model = create_dual_attention_whisper("openai/whisper-medium")  # 769M params
model = create_dual_attention_whisper("openai/whisper-large")   # 1550M params
```

### 2. Freezing Strategies
```python
# Strategy 1: Freeze encoder only
model.freeze_encoder()

# Strategy 2: Freeze encoder + primary attention
model.freeze_encoder()
model.freeze_primary_decoder()

# Strategy 3: Full fine-tuning
# (no freezing)
```

### 3. Data Augmentation
```python
# Enable noise injection during training
collator = DataCollatorSpeechSeq2SeqWithPaddingAndNoise(
    processor=processor,
    add_noise=True,
    noise_prob=0.5,
    noise_level=0.01
)
```

## 📈 Performance Optimization

### GPU Utilization
- **Batch Size**: 16 per GPU (adjust based on VRAM)
- **Gradient Accumulation**: 1 (increase if OOM)
- **Mixed Precision**: FP16 + TF32 enabled
- **Data Loading**: 16 workers with persistent workers

### Speed Benchmarks (estimated)
- **whisper-small** on 1x RTX 3090: ~100 samples/sec
- **whisper-medium** on 2x RTX 3090: ~80 samples/sec
- **whisper-large** on 4x A100: ~120 samples/sec

## 🧪 Evaluation Metrics

### Primary Metrics
- **WER (Word Error Rate)**: Main metric (lower is better)
- **CER (Character Error Rate)**: Secondary metric

### Evaluation Process
```bash
python scripts/evaluate.py \
    --model_path ./outputs/best_model \
    --test_data ./data/test.json \
    --output_file results.txt
```

## 🚀 Deployment

### Option 1: Python API
```python
from src.model.dual_whisper import DualAttentionWhisperForConditionalGeneration
model = DualAttentionWhisperForConditionalGeneration.from_pretrained("./outputs")
```

### Option 2: Command Line
```bash
python scripts/inference.py --model_path ./outputs --audio_path audio.wav
```

### Option 3: Export to ONNX (Future Work)
```python
# TODO: Add ONNX export for production deployment
```

## 🔬 Future Improvements

1. **Multi-scale Encoder**: Use encoder outputs at different layers
2. **Learnable Noise Embeddings**: Explicit noise type modeling
3. **Attention Visualization**: Tools to visualize primary vs secondary attention
4. **Streaming Support**: Real-time transcription with dual attention
5. **Quantization**: INT8 quantization for faster inference
6. **Knowledge Distillation**: Distill to smaller models

## 📝 Citation

If you use this work, please cite:

```bibtex
@article{dual_attention_whisper_2025,
  title={Dual-Attention Mechanism for Noise-Robust Whisper-Based Speech Recognition},
  author={Abbaszade, Fuad},
  year={2025}
}
```

## 📞 Support

For questions or issues:
1. Check QUICKSTART.md
2. Review example notebooks
3. Open an issue on GitHub

## 🎓 Learning Resources

- **Transformers**: https://huggingface.co/docs/transformers
- **Whisper Paper**: https://arxiv.org/abs/2212.04356
- **Attention Mechanisms**: https://arxiv.org/abs/1706.03762

---

**Built with ❤️ for noise-robust speech recognition**
