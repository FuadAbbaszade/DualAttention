"""
Dual-Attention Decoder for Whisper.

This module implements a modified Whisper decoder with dual cross-attention:
1. Primary cross-attention for linguistic content (speech)
2. Secondary cross-attention for noise regions

Both attend to the same encoder output but learn different attention patterns.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from transformers.models.whisper.modeling_whisper import (
    WhisperAttention,
    WhisperDecoderLayer,
    WhisperDecoder,
)


class DualCrossAttentionGate(nn.Module):
    """
    Gating mechanism to fuse primary and secondary cross-attention outputs.
    
    The gate learns to weight the two attention outputs dynamically:
    - High gate value: Trust primary attention (clean speech)
    - Low gate value: Use secondary attention (noise-aware)
    """
    
    def __init__(self, hidden_size: int):
        super().__init__()
        # Gate network: learns to weight primary vs secondary attention
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),  # [query; primary; secondary]
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        query: torch.Tensor,
        primary_output: torch.Tensor,
        secondary_output: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse primary and secondary attention outputs.
        
        Args:
            query: Decoder hidden states [batch, seq_len, hidden]
            primary_output: Primary attention output [batch, seq_len, hidden]
            secondary_output: Secondary attention output [batch, seq_len, hidden]
            
        Returns:
            Fused attention output [batch, seq_len, hidden]
        """
        # Concatenate for gate input
        gate_input = torch.cat([query, primary_output, secondary_output], dim=-1)
        
        # Compute gate value (0 to 1)
        alpha = self.gate(gate_input)  # [batch, seq_len, 1]
        
        # Weighted fusion: alpha * primary + (1 - alpha) * secondary
        # High alpha -> more primary (clean speech)
        # Low alpha -> more secondary (noise-aware)
        fused = alpha * primary_output + (1 - alpha) * secondary_output
        
        return fused


class DualAttentionDecoderLayer(WhisperDecoderLayer):
    """
    Modified Whisper decoder layer with dual cross-attention mechanism.
    
    Architecture:
        Input → Self-Attention → Primary Cross-Attn (speech) ──┐
                               → Secondary Cross-Attn (noise) ──┼→ Gate → FFN → Output
    """
    
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx=layer_idx)
        
        # Add secondary cross-attention head (same architecture as primary)
        self.encoder_attn_secondary = WhisperAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            is_causal=False,
            layer_idx=layer_idx,
            config=config,
        )
        
        # Layer norm for secondary cross-attention
        self.encoder_attn_secondary_layer_norm = nn.LayerNorm(self.embed_dim)
        
        # Gating mechanism to fuse primary and secondary outputs
        self.cross_attn_gate = DualCrossAttentionGate(self.embed_dim)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Forward pass with dual cross-attention.
        
        Args:
            hidden_states: Decoder input [batch, seq_len, hidden]
            attention_mask: Causal mask for self-attention
            encoder_hidden_states: Encoder output [batch, enc_seq_len, hidden]
            encoder_attention_mask: Mask for encoder outputs
            past_key_values: Cached key/values for generation
            cache_position: Position in cache
            ...
            
        Returns:
            (hidden_states, present_key_value, all_attentions)
        """
        residual = hidden_states
        
        # 1. Self-Attention (standard)
        hidden_states = self.self_attn_layer_norm(hidden_states)
        
        # Prepare self-attention kwargs
        self_attn_kwargs = {
            "hidden_states": hidden_states,
            "attention_mask": attention_mask,
            "layer_head_mask": layer_head_mask,
            "output_attentions": output_attentions,
        }
        if past_key_values is not None:
            self_attn_kwargs["past_key_values"] = past_key_values
        if cache_position is not None:
            self_attn_kwargs["cache_position"] = cache_position
            
        # WhisperAttention returns (hidden_states, attn_weights) in newer versions
        attn_output = self.self_attn(**self_attn_kwargs)
        if len(attn_output) == 2:
            hidden_states, self_attn_weights = attn_output
            present_key_value = None
        else:
            hidden_states, self_attn_weights, present_key_value = attn_output
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states
        
        # 2. Cross-Attention (DUAL)
        if encoder_hidden_states is not None:
            residual = hidden_states
            
            # 2a. Primary Cross-Attention (for speech)
            hidden_states_primary = self.encoder_attn_layer_norm(hidden_states)
            primary_attn_output = self.encoder_attn(
                hidden_states=hidden_states_primary,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                output_attentions=output_attentions,
            )
            if len(primary_attn_output) == 2:
                hidden_states_primary, cross_attn_weights_primary = primary_attn_output
            else:
                hidden_states_primary, cross_attn_weights_primary, _ = primary_attn_output
            hidden_states_primary = nn.functional.dropout(
                hidden_states_primary, p=self.dropout, training=self.training
            )
            
            # 2b. Secondary Cross-Attention (for noise)
            hidden_states_secondary = self.encoder_attn_secondary_layer_norm(hidden_states)
            secondary_attn_output = self.encoder_attn_secondary(
                hidden_states=hidden_states_secondary,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                output_attentions=output_attentions,
            )
            if len(secondary_attn_output) == 2:
                hidden_states_secondary, cross_attn_weights_secondary = secondary_attn_output
            else:
                hidden_states_secondary, cross_attn_weights_secondary, _ = secondary_attn_output
            hidden_states_secondary = nn.functional.dropout(
                hidden_states_secondary, p=self.dropout, training=self.training
            )
            
            # 2c. Fuse primary and secondary attention via gating
            hidden_states_fused = self.cross_attn_gate(
                query=hidden_states,
                primary_output=hidden_states_primary,
                secondary_output=hidden_states_secondary,
            )
            
            hidden_states = residual + hidden_states_fused
        else:
            cross_attn_weights_primary = None
            cross_attn_weights_secondary = None
        
        # 3. Feed-Forward Network (standard)
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.activation_dropout, training=self.training
        )
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,)
        
        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights_primary, cross_attn_weights_secondary)
            
        if use_cache:
            outputs += (present_key_value,)
            
        return outputs


class DualAttentionWhisperDecoder(WhisperDecoder):
    """
    Modified Whisper decoder with dual cross-attention layers.
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # Replace standard decoder layers with dual-attention layers
        self.layers = nn.ModuleList(
            [DualAttentionDecoderLayer(config, layer_idx=i) for i in range(config.decoder_layers)]
        )
        
        # Re-initialize weights (important!)
        self.post_init()
