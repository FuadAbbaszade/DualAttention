"""Data collator for batching audio samples with padding."""

import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator that pads audio features and labels for batching.
    
    Handles:
    - Padding input features to same length
    - Padding labels and replacing padding with -100 (ignored in loss)
    - Creating attention masks
    """
    
    processor: Any
    decoder_start_token_id: int = None
    
    def __post_init__(self):
        if self.decoder_start_token_id is None:
            self.decoder_start_token_id = self.processor.tokenizer.bos_token_id
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of samples.
        
        Args:
            features: List of dictionaries with "input_features" and "labels"
            
        Returns:
            Batch dictionary with padded tensors
        """
        # Extract input features and labels
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        labels = [feature["labels"] for feature in features]
        
        # Pad input features (Mel spectrograms)
        # Whisper expects [batch, 80, 3000] - features are already this shape
        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt"
        )
        
        # Pad labels (token sequences)
        # Find max length in batch
        max_label_length = max(len(label) for label in labels)
        
        # Pad each label sequence
        padded_labels = []
        for label in labels:
            # Pad with -100 (ignored in cross-entropy loss)
            padding_length = max_label_length - len(label)
            padded_label = torch.cat([
                label,
                torch.full((padding_length,), -100, dtype=label.dtype)
            ])
            padded_labels.append(padded_label)
        
        batch["labels"] = torch.stack(padded_labels)
        
        return batch


@dataclass  
class DataCollatorSpeechSeq2SeqWithPaddingAndNoise:
    """
    Advanced data collator with online noise augmentation.
    
    This collator can add noise during training for better robustness.
    """
    
    processor: Any
    decoder_start_token_id: int = None
    add_noise: bool = True
    noise_prob: float = 0.5
    noise_level: float = 0.01
    
    def __post_init__(self):
        if self.decoder_start_token_id is None:
            self.decoder_start_token_id = self.processor.tokenizer.bos_token_id
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        """Collate with optional noise augmentation."""
        
        # Extract and potentially augment input features
        input_features = []
        for feature in features:
            feat = feature["input_features"]
            
            # Add noise augmentation during training
            if self.add_noise and torch.rand(1).item() < self.noise_prob:
                noise = torch.randn_like(feat) * self.noise_level
                feat = feat + noise
            
            input_features.append({"input_features": feat})
        
        labels = [feature["labels"] for feature in features]
        
        # Pad features
        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt"
        )
        
        # Pad labels
        max_label_length = max(len(label) for label in labels)
        padded_labels = []
        for label in labels:
            padding_length = max_label_length - len(label)
            padded_label = torch.cat([
                label,
                torch.full((padding_length,), -100, dtype=label.dtype)
            ])
            padded_labels.append(padded_label)
        
        batch["labels"] = torch.stack(padded_labels)
        
        return batch
