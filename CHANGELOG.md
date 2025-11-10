# Changelog

All notable changes to the Dual-Attention Whisper project.

## [Unreleased]

### Added
- **Hugging Face Dataset Integration**: Direct support for loading datasets from Hugging Face Hub
  - Added `--hf_dataset`, `--hf_config`, `--hf_split` arguments to `prepare_data.py`
  - Automatic dataset caching and preprocessing
  - Support for both local files and HF datasets
- **Improved GPU Memory Management**: Documented memory requirements for all model sizes
- **API Compatibility**: Updated for latest `transformers` library (4.30+)
  - Fixed `past_key_values` parameter handling
  - Added `cache_position` support
  - Added `layer_idx` to all attention modules
  - Fixed return value unpacking for newer WhisperAttention API

### Changed
- **Script Rename**: `scripts/evaluate.py` → `scripts/run_evaluation.py` to avoid name collision with `evaluate` package
- **Import Organization**: Moved all imports to module level in `prepare_data.py` for better performance
- **Training Defaults**: Updated recommended batch sizes based on empirical testing on A100 40GB
  - whisper-small: batch 16 (previously: varied)
  - whisper-medium: batch 12 (previously: untested, batch 28 causes OOM)
  - Added GPU memory requirement tables

### Fixed
- **Circular Import**: Fixed circular import between `evaluate` package and `scripts/evaluate.py`
- **Audio Loading**: Fixed audio loading from Hugging Face datasets
  - Handle `bytes`, `path`, and `array` formats
  - Use dataset filesystem for relative paths
  - Proper error handling for missing audio data
- **UnboundLocalError**: Fixed scoping issues with `soundfile`, `numpy`, `io`, and `random` imports
- **Model Compatibility**: Fixed compatibility with transformers 4.30+
  - Updated `DualAttentionDecoderLayer` forward signature
  - Added config parameter to secondary attention module
  - Fixed attention output unpacking (2 vs 3 values)

### Documentation
- **README.md**: Complete rewrite with Hugging Face support, GPU requirements, and badges
- **USAGE_GUIDE.md**: Added comprehensive HF dataset instructions and GPU memory tables
- **Examples**: Updated all code examples to use new HF dataset integration

## [1.0.0] - Initial Release

### Added
- Dual-attention mechanism for Whisper decoder
- Primary and secondary cross-attention branches
- Gating mechanism for attention fusion
- Training pipeline with optimized settings
- Evaluation metrics (WER, CER)
- Inference scripts
- Data preparation utilities
- Multi-GPU support with DDP
- FP16 + TF32 mixed precision training
- Frozen encoder training option

### Features
- Compatible with all Whisper model sizes (tiny to large)
- Preserves pre-trained Whisper weights
- Noise-robust speech recognition
- Support for low-resource languages (tested on Azerbaijani)
