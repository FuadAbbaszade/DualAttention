#!/usr/bin/env python3
"""
Visualization script for dual attention weights.

This script helps visualize the difference between primary and secondary attention patterns.
"""

import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformers import WhisperProcessor
from model.dual_whisper import DualAttentionWhisperForConditionalGeneration
import librosa


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Visualize dual attention patterns")
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model"
    )
    parser.add_argument(
        "--audio_path",
        type=str,
        required=True,
        help="Path to audio file"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="az",
        help="Language code"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./attention_visualizations",
        help="Output directory for visualizations"
    )
    parser.add_argument(
        "--layer_idx",
        type=int,
        default=0,
        help="Decoder layer to visualize (0-indexed)"
    )
    
    return parser.parse_args()


def extract_attention_weights(model, processor, audio_path, language):
    """
    Extract primary and secondary attention weights from the model.
    
    Note: This requires modifications to the model to return attention weights.
    This is a template implementation.
    """
    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000)
    
    # Extract features
    input_features = processor.feature_extractor(
        audio,
        sampling_rate=16000,
        return_tensors="pt"
    ).input_features
    
    # Generate with attention output
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            input_features,
            language=language,
            task="transcribe",
            output_attentions=True,
            return_dict_in_generate=True,
            max_length=50
        )
    
    # Extract attention weights
    # Note: This is a simplified version. Actual implementation depends on
    # how attention weights are stored in the model outputs
    
    print("⚠️  Attention weight extraction requires model modifications")
    print("This is a template for future implementation")
    
    return None, None, outputs


def plot_attention_comparison(
    primary_attn,
    secondary_attn,
    tokens,
    output_path
):
    """
    Plot side-by-side comparison of primary vs secondary attention.
    
    Args:
        primary_attn: Primary attention weights [num_heads, seq_len, enc_len]
        secondary_attn: Secondary attention weights [num_heads, seq_len, enc_len]
        tokens: Decoded tokens
        output_path: Where to save the plot
    """
    if primary_attn is None or secondary_attn is None:
        print("⚠️  Cannot plot: attention weights not available")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot primary attention
    sns.heatmap(
        primary_attn[0].cpu().numpy(),  # First head
        ax=axes[0],
        cmap="viridis",
        xticklabels=False,
        yticklabels=tokens,
        cbar_kws={"label": "Attention Weight"}
    )
    axes[0].set_title("Primary Attention (Speech)")
    axes[0].set_xlabel("Encoder Timesteps")
    axes[0].set_ylabel("Decoder Tokens")
    
    # Plot secondary attention
    sns.heatmap(
        secondary_attn[0].cpu().numpy(),  # First head
        ax=axes[1],
        cmap="magma",
        xticklabels=False,
        yticklabels=tokens,
        cbar_kws={"label": "Attention Weight"}
    )
    axes[1].set_title("Secondary Attention (Noise)")
    axes[1].set_xlabel("Encoder Timesteps")
    axes[1].set_ylabel("Decoder Tokens")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ Saved visualization to {output_path}")
    plt.close()


def plot_gate_values(gate_values, tokens, output_path):
    """
    Plot the gating values (alpha) over time.
    
    High alpha = trusts primary attention (speech)
    Low alpha = trusts secondary attention (noise)
    """
    if gate_values is None:
        print("⚠️  Cannot plot: gate values not available")
        return
    
    plt.figure(figsize=(12, 4))
    plt.plot(gate_values, marker='o', linewidth=2)
    plt.fill_between(range(len(gate_values)), gate_values, alpha=0.3)
    plt.axhline(y=0.5, color='r', linestyle='--', label='Equal weighting')
    
    plt.title("Dual-Attention Gate Values Over Time")
    plt.xlabel("Token Position")
    plt.ylabel("Gate Value (α)")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add token labels
    if tokens:
        plt.xticks(range(len(tokens)), tokens, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ Saved gate visualization to {output_path}")
    plt.close()


def plot_attention_difference(primary_attn, secondary_attn, tokens, output_path):
    """
    Plot the difference between primary and secondary attention.
    
    Shows where the two attention mechanisms focus differently.
    """
    if primary_attn is None or secondary_attn is None:
        return
    
    diff = primary_attn[0].cpu().numpy() - secondary_attn[0].cpu().numpy()
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(
        diff,
        cmap="RdBu_r",
        center=0,
        xticklabels=False,
        yticklabels=tokens,
        cbar_kws={"label": "Attention Difference"}
    )
    plt.title("Attention Difference (Primary - Secondary)")
    plt.xlabel("Encoder Timesteps")
    plt.ylabel("Decoder Tokens")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ Saved difference visualization to {output_path}")
    plt.close()


def main():
    """Main visualization function."""
    args = parse_args()
    
    print("\n" + "="*70)
    print("🎨 DUAL-ATTENTION VISUALIZATION")
    print("="*70)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"\n📦 Loading model from {args.model_path}...")
    processor = WhisperProcessor.from_pretrained(args.model_path)
    model = DualAttentionWhisperForConditionalGeneration.from_pretrained(args.model_path)
    model.eval()
    
    # Extract attention weights
    print(f"\n🎵 Processing audio: {args.audio_path}")
    primary_attn, secondary_attn, outputs = extract_attention_weights(
        model, processor, args.audio_path, args.language
    )
    
    # Decode transcription
    transcription = processor.batch_decode(
        outputs.sequences,
        skip_special_tokens=True
    )[0]
    
    tokens = processor.tokenizer.tokenize(transcription)
    
    print(f"\n📝 Transcription: {transcription}")
    print(f"📊 Tokens: {tokens}")
    
    # Create visualizations
    print("\n🎨 Creating visualizations...")
    
    audio_name = Path(args.audio_path).stem
    
    # 1. Side-by-side attention comparison
    plot_attention_comparison(
        primary_attn,
        secondary_attn,
        tokens,
        output_dir / f"{audio_name}_attention_comparison.png"
    )
    
    # 2. Gate values (if available)
    # plot_gate_values(gate_values, tokens, output_dir / f"{audio_name}_gate_values.png")
    
    # 3. Attention difference
    plot_attention_difference(
        primary_attn,
        secondary_attn,
        tokens,
        output_dir / f"{audio_name}_attention_difference.png"
    )
    
    print("\n✅ Visualization complete!")
    print(f"📁 Output directory: {output_dir}")
    print("\n" + "="*70)
    
    print("\n⚠️  NOTE: Full attention visualization requires model modifications")
    print("To enable detailed attention extraction:")
    print("1. Modify DualAttentionDecoderLayer to store attention weights")
    print("2. Return attention weights in model.generate()")
    print("3. Extract weights in this script")
    print("\nThis is left as an exercise for advanced users.")


if __name__ == "__main__":
    main()
