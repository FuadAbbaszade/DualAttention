#!/usr/bin/env python3
"""
Inference script for Dual-Attention Whisper.

Run transcription on audio files using trained model.
"""

import sys
import argparse
import torch
from pathlib import Path
import librosa
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformers import WhisperProcessor
from model.dual_whisper import DualAttentionWhisperForConditionalGeneration


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run inference with Dual-Attention Whisper")
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--audio_path",
        type=str,
        required=True,
        help="Path to audio file or directory"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="az",
        help="Language code (e.g., 'az' for Azerbaijani)"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="transcribe",
        choices=["transcribe", "translate"],
        help="Task: transcribe or translate to English"
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=5,
        help="Number of beams for beam search"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output file to save transcriptions"
    )
    
    return parser.parse_args()


def load_audio(audio_path: str, sampling_rate: int = 16000):
    """Load and resample audio file."""
    audio, sr = librosa.load(audio_path, sr=sampling_rate, mono=True)
    return audio


def transcribe_audio(
    model,
    processor,
    audio_path: str,
    language: str = "az",
    task: str = "transcribe",
    num_beams: int = 5,
    device: str = "cuda",
):
    """
    Transcribe a single audio file.
    
    Args:
        model: Trained model
        processor: WhisperProcessor
        audio_path: Path to audio file
        language: Language code
        task: "transcribe" or "translate"
        num_beams: Beam search width
        device: Device to run on
        
    Returns:
        Transcription text and processing time
    """
    # Load audio
    audio = load_audio(audio_path)
    
    # Extract features
    input_features = processor.feature_extractor(
        audio,
        sampling_rate=16000,
        return_tensors="pt"
    ).input_features.to(device)
    
    # Generate transcription
    start_time = time.time()
    
    with torch.no_grad():
        generated_ids = model.generate(
            input_features,
            max_length=448,
            num_beams=num_beams,
            language=language,
            task=task,
        )
    
    elapsed_time = time.time() - start_time
    
    # Decode
    transcription = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]
    
    return transcription, elapsed_time


def main():
    """Main inference function."""
    args = parse_args()
    
    print("\n" + "="*70)
    print("🎤 DUAL-ATTENTION WHISPER INFERENCE")
    print("="*70)
    
    # Load model and processor
    print(f"\n📦 Loading model from {args.model_path}...")
    processor = WhisperProcessor.from_pretrained(args.model_path)
    model = DualAttentionWhisperForConditionalGeneration.from_pretrained(args.model_path)
    model.to(args.device)
    model.eval()
    print("✅ Model loaded")
    
    # Get audio files
    audio_path = Path(args.audio_path)
    if audio_path.is_file():
        audio_files = [audio_path]
    elif audio_path.is_dir():
        # Find all audio files
        audio_files = list(audio_path.glob("*.wav")) + \
                     list(audio_path.glob("*.mp3")) + \
                     list(audio_path.glob("*.flac"))
    else:
        raise ValueError(f"Invalid audio path: {args.audio_path}")
    
    print(f"\n🎵 Found {len(audio_files)} audio file(s)")
    
    # Transcribe each file
    results = []
    total_time = 0
    
    print("\n" + "="*70)
    print("📝 TRANSCRIPTIONS")
    print("="*70)
    
    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n[{i}/{len(audio_files)}] {audio_file.name}")
        
        try:
            transcription, elapsed = transcribe_audio(
                model=model,
                processor=processor,
                audio_path=str(audio_file),
                language=args.language,
                task=args.task,
                num_beams=args.num_beams,
                device=args.device,
            )
            
            print(f"  Time: {elapsed:.2f}s")
            print(f"  Text: {transcription}")
            
            results.append({
                "file": str(audio_file),
                "transcription": transcription,
                "time": elapsed,
            })
            
            total_time += elapsed
            
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
    
    # Summary
    print("\n" + "="*70)
    print("📊 SUMMARY")
    print("="*70)
    print(f"Total files: {len(audio_files)}")
    print(f"Successful: {len(results)}")
    print(f"Total time: {total_time:.2f}s")
    if results:
        print(f"Average time: {total_time/len(results):.2f}s per file")
    print("="*70)
    
    # Save to file
    if args.output_file:
        output_path = Path(args.output_file)
        with open(output_path, "w", encoding="utf-8") as f:
            for result in results:
                f.write(f"{result['file']}\t{result['transcription']}\n")
        print(f"\n💾 Transcriptions saved to: {output_path}")
    
    print()


if __name__ == "__main__":
    main()
