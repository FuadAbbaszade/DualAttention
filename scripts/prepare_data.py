#!/usr/bin/env python3
"""
Data preparation script for Dual-Attention Whisper.

Convert raw audio files and transcriptions into the required JSON format.
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict
import librosa
from tqdm import tqdm


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Prepare data for Dual-Attention Whisper")
    
    parser.add_argument(
        "--audio_dir",
        type=str,
        required=True,
        help="Directory containing audio files"
    )
    parser.add_argument(
        "--transcripts",
        type=str,
        required=True,
        help="Path to transcripts file (JSON or TXT)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/processed",
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.9,
        help="Ratio for train/eval split"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="az",
        help="Language code"
    )
    parser.add_argument(
        "--min_duration",
        type=float,
        default=0.5,
        help="Minimum audio duration in seconds"
    )
    parser.add_argument(
        "--max_duration",
        type=float,
        default=30.0,
        help="Maximum audio duration in seconds"
    )
    
    return parser.parse_args()


def load_transcripts(transcript_path: str) -> Dict[str, str]:
    """
    Load transcripts from file.
    
    Supported formats:
    1. JSON: {"filename": "transcription", ...}
    2. TXT: filename\ttranscription (one per line)
    """
    transcript_path = Path(transcript_path)
    
    if transcript_path.suffix == ".json":
        with open(transcript_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    elif transcript_path.suffix == ".txt":
        transcripts = {}
        with open(transcript_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    filename, text = parts
                    transcripts[filename] = text
        return transcripts
    
    else:
        raise ValueError(f"Unsupported transcript format: {transcript_path.suffix}")


def get_audio_duration(audio_path: str) -> float:
    """Get audio duration in seconds."""
    try:
        duration = librosa.get_duration(path=audio_path)
        return duration
    except Exception as e:
        print(f"Warning: Could not get duration for {audio_path}: {e}")
        return 0.0


def prepare_dataset(
    audio_dir: str,
    transcripts: Dict[str, str],
    language: str,
    min_duration: float,
    max_duration: float,
) -> List[Dict]:
    """
    Prepare dataset from audio files and transcripts.
    
    Returns:
        List of dictionaries with audio_path, text, language, duration
    """
    audio_dir = Path(audio_dir)
    dataset = []
    
    # Find all audio files
    audio_extensions = [".wav", ".mp3", ".flac", ".m4a"]
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(audio_dir.glob(f"**/*{ext}"))
    
    print(f"Found {len(audio_files)} audio files")
    
    # Process each audio file
    for audio_path in tqdm(audio_files, desc="Processing audio files"):
        # Get filename without extension
        filename = audio_path.stem
        
        # Check if transcript exists
        # Try multiple keys: with extension, without extension, relative path
        text = None
        for key in [audio_path.name, filename, str(audio_path.relative_to(audio_dir))]:
            if key in transcripts:
                text = transcripts[key]
                break
        
        if text is None:
            print(f"Warning: No transcript found for {audio_path.name}")
            continue
        
        # Get duration
        duration = get_audio_duration(str(audio_path))
        
        # Filter by duration
        if duration < min_duration or duration > max_duration:
            continue
        
        # Add to dataset
        dataset.append({
            "audio_path": str(audio_path.absolute()),
            "text": text,
            "language": language,
            "duration": duration,
        })
    
    return dataset


def main():
    """Main data preparation function."""
    args = parse_args()
    
    print("\n" + "="*70)
    print("📊 DATA PREPARATION FOR DUAL-ATTENTION WHISPER")
    print("="*70)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load transcripts
    print(f"\n📝 Loading transcripts from {args.transcripts}...")
    transcripts = load_transcripts(args.transcripts)
    print(f"✅ Loaded {len(transcripts)} transcripts")
    
    # Prepare dataset
    print(f"\n🎵 Processing audio files from {args.audio_dir}...")
    dataset = prepare_dataset(
        audio_dir=args.audio_dir,
        transcripts=transcripts,
        language=args.language,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
    )
    
    print(f"\n✅ Processed {len(dataset)} samples")
    
    # Split into train/eval
    import random
    random.shuffle(dataset)
    
    split_idx = int(len(dataset) * args.train_split)
    train_dataset = dataset[:split_idx]
    eval_dataset = dataset[split_idx:]
    
    print(f"\n📊 Dataset split:")
    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Eval:  {len(eval_dataset)} samples")
    
    # Save datasets
    train_path = output_dir / "train.json"
    eval_path = output_dir / "eval.json"
    
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_dataset, f, ensure_ascii=False, indent=2)
    
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_dataset, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Saved datasets:")
    print(f"   Train: {train_path}")
    print(f"   Eval:  {eval_path}")
    
    # Print statistics
    if train_dataset:
        durations = [item["duration"] for item in train_dataset]
        print(f"\n📈 Train dataset statistics:")
        print(f"   Total duration: {sum(durations)/3600:.2f} hours")
        print(f"   Average duration: {sum(durations)/len(durations):.2f}s")
        print(f"   Min duration: {min(durations):.2f}s")
        print(f"   Max duration: {max(durations):.2f}s")
    
    print("\n✅ Data preparation complete!\n")


if __name__ == "__main__":
    main()
