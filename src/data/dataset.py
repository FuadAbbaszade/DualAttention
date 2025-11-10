"""
Dataset for noisy audio with transcriptions.

Supports various audio formats and handles:
- Audio loading and resampling
- Feature extraction
- Text tokenization
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
from typing import Dict, List, Optional, Union
import librosa
import numpy as np


class NoisyAudioDataset(Dataset):
    """
    Dataset for training Dual-Attention Whisper on noisy audio.
    
    Data format:
        {
            "audio_path": "path/to/audio.wav",
            "text": "ground truth transcription",
            "language": "az",  # optional
            "duration": 5.2     # optional
        }
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        processor,
        max_audio_length: float = 30.0,  # seconds
        sampling_rate: int = 16000,
        language: Optional[str] = None,
        task: str = "transcribe",
    ):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to JSON file with dataset
            processor: WhisperProcessor instance
            max_audio_length: Maximum audio length in seconds
            sampling_rate: Target sampling rate (16kHz for Whisper)
            language: Force language (e.g., "az" for Azerbaijani)
            task: "transcribe" or "translate"
        """
        self.processor = processor
        self.max_audio_length = max_audio_length
        self.sampling_rate = sampling_rate
        self.language = language
        self.task = task
        
        # Load dataset
        data_path = Path(data_path)
        if data_path.suffix == ".json":
            with open(data_path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        elif data_path.is_dir():
            # Load all JSON files in directory
            self.data = []
            for json_file in data_path.glob("*.json"):
                with open(json_file, "r", encoding="utf-8") as f:
                    self.data.extend(json.load(f))
        else:
            raise ValueError(f"Invalid data_path: {data_path}")
        
        print(f"✅ Loaded {len(self.data)} audio samples")
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            {
                "input_features": Mel spectrogram features [80, 3000]
                "labels": Token IDs for transcription
                "input_length": Actual audio length in frames
            }
        """
        item = self.data[idx]
        
        # Load audio
        audio_path = item["audio_path"]
        audio, sr = librosa.load(audio_path, sr=self.sampling_rate, mono=True)
        
        # Trim or pad to max_audio_length
        max_samples = int(self.max_audio_length * self.sampling_rate)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        
        # Extract Mel spectrogram features
        input_features = self.processor.feature_extractor(
            audio,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
        ).input_features[0]  # [80, 3000]
        
        # Tokenize text
        text = item["text"]
        
        # Prepare prompt tokens (language + task)
        lang = item.get("language", self.language)
        forced_decoder_ids = []
        
        if lang:
            lang_token = f"<|{lang}|>"
            forced_decoder_ids.append(self.processor.tokenizer.convert_tokens_to_ids(lang_token))
        
        task_token = f"<|{self.task}|>"
        forced_decoder_ids.append(self.processor.tokenizer.convert_tokens_to_ids(task_token))
        
        # Tokenize transcription
        labels = self.processor.tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=448,  # Whisper max length
        ).input_ids[0]
        
        # Add forced decoder IDs at the start
        if forced_decoder_ids:
            labels = torch.cat([
                torch.tensor(forced_decoder_ids, dtype=labels.dtype),
                labels
            ])
        
        return {
            "input_features": input_features,
            "labels": labels,
            "input_length": len(audio) / self.sampling_rate,
        }
    
    def filter_by_duration(
        self,
        min_duration: float = 0.5,
        max_duration: Optional[float] = None,
    ):
        """Filter dataset by audio duration."""
        if max_duration is None:
            max_duration = self.max_audio_length
            
        original_len = len(self.data)
        self.data = [
            item for item in self.data
            if min_duration <= item.get("duration", float("inf")) <= max_duration
        ]
        print(f"Filtered: {original_len} → {len(self.data)} samples")


class AudioAugmentation:
    """
    Simple audio augmentation for training robustness.
    
    You can add:
    - Background noise injection
    - Speed perturbation
    - Volume changes
    - Time masking
    """
    
    def __init__(
        self,
        add_noise: bool = True,
        noise_level: float = 0.005,
        speed_perturb: bool = True,
        speed_range: tuple = (0.9, 1.1),
    ):
        self.add_noise = add_noise
        self.noise_level = noise_level
        self.speed_perturb = speed_perturb
        self.speed_range = speed_range
    
    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """Apply augmentation to audio."""
        # Add Gaussian noise
        if self.add_noise and np.random.rand() < 0.5:
            noise = np.random.randn(len(audio)) * self.noise_level
            audio = audio + noise
        
        # Speed perturbation
        if self.speed_perturb and np.random.rand() < 0.5:
            speed = np.random.uniform(*self.speed_range)
            audio = librosa.effects.time_stretch(audio, rate=speed)
        
        return audio
