"""
Complete Dual-Attention Whisper Model.

This module provides the full model that combines:
- Standard Whisper encoder (unchanged)
- Modified dual-attention decoder
"""

import torch
import torch.nn as nn
from typing import Optional, Union, Tuple
from transformers import WhisperForConditionalGeneration, WhisperConfig
from transformers.modeling_outputs import Seq2SeqLMOutput

from .dual_attention_decoder import DualAttentionWhisperDecoder


class DualAttentionWhisperForConditionalGeneration(WhisperForConditionalGeneration):
    """
    Whisper model with dual cross-attention decoder for noise-robust ASR.
    
    Architecture:
        Audio → Encoder (unchanged) → Dual-Attention Decoder → Text
        
    The encoder remains unchanged to preserve pre-trained weights.
    Only the decoder is modified with dual cross-attention.
    """
    
    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        
        # Replace the standard decoder with dual-attention decoder
        self.model.decoder = DualAttentionWhisperDecoder(config)
        
        # Re-initialize new parameters
        self.post_init()
        
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args,
        **kwargs
    ):
        """
        Load a pre-trained Whisper model and convert it to dual-attention.
        
        This method:
        1. Loads standard Whisper weights
        2. Initializes dual-attention decoder
        3. Copies primary attention weights from pre-trained model
        4. Randomly initializes secondary attention and gate
        
        Args:
            pretrained_model_name_or_path: Hugging Face model name or local path
                                          e.g., "openai/whisper-small"
        
        Returns:
            DualAttentionWhisperForConditionalGeneration instance
        """
        # Load standard Whisper model first
        standard_model = WhisperForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            **kwargs
        )
        
        # Create dual-attention model with same config
        config = standard_model.config
        dual_model = cls(config)
        
        # Copy ALL weights from standard model
        dual_model.load_state_dict(standard_model.state_dict(), strict=False)
        
        # Initialize secondary attention from primary attention
        # This gives a warm start for the secondary attention head
        for layer_idx in range(config.decoder_layers):
            dual_layer = dual_model.model.decoder.layers[layer_idx]
            
            # Copy primary cross-attention weights to secondary
            dual_layer.encoder_attn_secondary.load_state_dict(
                dual_layer.encoder_attn.state_dict()
            )
            
            # Copy layer norm weights
            dual_layer.encoder_attn_secondary_layer_norm.load_state_dict(
                dual_layer.encoder_attn_layer_norm.state_dict()
            )
            
            # Gate is randomly initialized (learns from scratch)
            # This is intentional - the gate learns to balance speech vs noise
        
        print("✅ Loaded pre-trained Whisper and initialized dual-attention decoder")
        print(f"   - Encoder: Pre-trained weights (frozen recommended)")
        print(f"   - Decoder primary attention: Pre-trained weights")
        print(f"   - Decoder secondary attention: Copied from primary (will fine-tune)")
        print(f"   - Cross-attention gates: Randomly initialized (will learn)")
        
        return dual_model
    
    def freeze_encoder(self):
        """
        Freeze encoder parameters to preserve pre-trained representations.
        
        During fine-tuning, it's recommended to:
        1. Freeze encoder initially
        2. Train decoder, secondary attention, and gates
        3. Optionally unfreeze encoder for final fine-tuning
        """
        for param in self.model.encoder.parameters():
            param.requires_grad = False
        print("🔒 Encoder frozen")
        
    def unfreeze_encoder(self):
        """Unfreeze encoder for full model fine-tuning."""
        for param in self.model.encoder.parameters():
            param.requires_grad = True
        print("🔓 Encoder unfrozen")
        
    def freeze_primary_decoder(self):
        """
        Freeze primary decoder attention (keeps pre-trained Whisper behavior).
        Only train secondary attention and gates.
        """
        for layer in self.model.decoder.layers:
            # Freeze primary cross-attention
            for param in layer.encoder_attn.parameters():
                param.requires_grad = False
            for param in layer.encoder_attn_layer_norm.parameters():
                param.requires_grad = False
        print("🔒 Primary decoder attention frozen")
        
    def get_trainable_params_info(self):
        """Print information about trainable parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print("\n" + "="*60)
        print("📊 MODEL PARAMETERS")
        print("="*60)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Frozen parameters: {total_params - trainable_params:,}")
        print(f"Trainable ratio: {100 * trainable_params / total_params:.2f}%")
        print("="*60 + "\n")
        
        return {
            "total": total_params,
            "trainable": trainable_params,
            "frozen": total_params - trainable_params,
        }
    
    def forward(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], Seq2SeqLMOutput]:
        """
        Forward pass - same signature as standard Whisper.
        
        The dual-attention mechanism is transparent to the training loop.
        """
        return super().forward(
            input_features=input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


def create_dual_attention_whisper(
    model_name: str = "openai/whisper-small",
    freeze_encoder: bool = True,
    freeze_primary_decoder: bool = False,
) -> DualAttentionWhisperForConditionalGeneration:
    """
    Convenience function to create and configure a dual-attention Whisper model.
    
    Args:
        model_name: Pre-trained Whisper model to load
        freeze_encoder: Whether to freeze encoder (recommended for initial training)
        freeze_primary_decoder: Whether to freeze primary decoder attention
        
    Returns:
        Configured DualAttentionWhisperForConditionalGeneration
        
    Example:
        >>> model = create_dual_attention_whisper(
        ...     model_name="openai/whisper-small",
        ...     freeze_encoder=True
        ... )
        >>> model.get_trainable_params_info()
    """
    print(f"\n🚀 Creating Dual-Attention Whisper from {model_name}")
    
    # Load and convert model
    model = DualAttentionWhisperForConditionalGeneration.from_pretrained(model_name)
    
    # Apply freezing strategy
    if freeze_encoder:
        model.freeze_encoder()
    
    if freeze_primary_decoder:
        model.freeze_primary_decoder()
    
    # Print trainable parameters
    model.get_trainable_params_info()
    
    return model
