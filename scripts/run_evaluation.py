#!/usr/bin/env python3
"""
Evaluation script for Dual-Attention Whisper.

Evaluate model on test dataset and compute WER/CER metrics.
"""

import sys
import argparse
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformers import WhisperProcessor
from model.dual_whisper import DualAttentionWhisperForConditionalGeneration
from data.dataset import NoisyAudioDataset
from training.metrics import evaluate_model


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate Dual-Attention Whisper")
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--test_data",
        type=str,
        required=True,
        help="Path to test data JSON"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="az",
        help="Language code (e.g., 'az' for Azerbaijani)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run evaluation on"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=5,
        help="Number of beams for beam search"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output file to save detailed results"
    )
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()
    
    print("\n" + "="*70)
    print("📊 DUAL-ATTENTION WHISPER EVALUATION")
    print("="*70)
    
    # Load model and processor
    print(f"\n📦 Loading model from {args.model_path}...")
    processor = WhisperProcessor.from_pretrained(args.model_path)
    model = DualAttentionWhisperForConditionalGeneration.from_pretrained(args.model_path)
    model.to(args.device)
    model.eval()
    print("✅ Model loaded")
    
    # Load test dataset
    print(f"\n📊 Loading test dataset from {args.test_data}...")
    test_dataset = NoisyAudioDataset(
        data_path=args.test_data,
        processor=processor,
        language=args.language,
    )
    print(f"✅ Loaded {len(test_dataset)} test samples")
    
    # Evaluate
    print("\n🔍 Running evaluation...")
    results = evaluate_model(
        model=model,
        processor=processor,
        test_dataset=test_dataset,
        device=args.device,
    )
    
    # Save detailed results
    if args.output_file:
        output_path = Path(args.output_file)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("="*70 + "\n")
            f.write("EVALUATION RESULTS\n")
            f.write("="*70 + "\n")
            f.write(f"WER: {results['wer']:.2f}%\n")
            f.write(f"CER: {results['cer']:.2f}%\n")
            f.write("="*70 + "\n\n")
            
            f.write("DETAILED PREDICTIONS\n")
            f.write("="*70 + "\n")
            for i, (pred, ref) in enumerate(zip(results['predictions'], results['references']), 1):
                f.write(f"\n[{i}]\n")
                f.write(f"Reference:  {ref}\n")
                f.write(f"Prediction: {pred}\n")
        
        print(f"\n💾 Detailed results saved to: {output_path}")
    
    print("\n✅ Evaluation complete!\n")


if __name__ == "__main__":
    main()
