#!/usr/bin/env python3
"""
Data preparation script for Dual-Attention Whisper.

Convert raw audio files and transcriptions into the required JSON format.
Supports both local files and Hugging Face datasets.
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm
import os
import io
import random


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Prepare data for Dual-Attention Whisper")
    
    # Data source arguments
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--audio_dir",
        type=str,
        help="Directory containing audio files (for local data)"
    )
    source_group.add_argument(
        "--hf_dataset",
        type=str,
        help="Hugging Face dataset name (e.g., 'mozilla-foundation/common_voice_13_0')"
    )
    
    parser.add_argument(
        "--hf_config",
        type=str,
        help="Hugging Face dataset config/subset (e.g., 'az' for Azerbaijani)"
    )
    parser.add_argument(
        "--hf_split",
        type=str,
        default="train",
        help="Hugging Face dataset split to use (default: train)"
    )
    parser.add_argument(
        "--hf_audio_column",
        type=str,
        default="audio",
        help="Column name for audio in HF dataset"
    )
    parser.add_argument(
        "--hf_text_column",
        type=str,
        default="sentence",
        help="Column name for text/transcription in HF dataset"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Directory to cache Hugging Face datasets (default: ~/.cache/huggingface)"
    )
    parser.add_argument(
        "--transcripts",
        type=str,
        help="Path to transcripts file (JSON or TXT) - only for local audio_dir"
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
        help="Ratio for train/eval split (only for local data)"
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
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (useful for testing)"
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


def prepare_hf_dataset(
    dataset_name: str,
    config: Optional[str],
    split: str,
    audio_column: str,
    text_column: str,
    language: str,
    min_duration: float,
    max_duration: float,
    cache_dir: Optional[str],
    max_samples: Optional[int],
    output_dir: Path,
) -> tuple:
    """
    Load and prepare dataset from Hugging Face.
    
    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    try:
        from datasets import load_dataset, Audio as HFAudio
    except ImportError:
        raise ImportError(
            "datasets library is required for Hugging Face datasets. "
            "Install it with: pip install datasets"
        )
    
    print(f"\n🤗 Loading Hugging Face dataset: {dataset_name}")
    if config:
        print(f"   Config: {config}")
    print(f"   Split: {split}")
    if cache_dir:
        print(f"   Cache directory: {cache_dir}")
    else:
        print(f"   Cache directory: ~/.cache/huggingface/datasets (default)")
    
    # Load dataset - it will cache automatically
    load_kwargs = {
        "path": dataset_name,
        "split": split,
    }
    if config:
        load_kwargs["name"] = config
    if cache_dir:
        load_kwargs["cache_dir"] = cache_dir
    
    print("\n⏳ Downloading dataset (only happens once, then cached)...")
    hf_dataset = load_dataset(**load_kwargs)

    # Disable automatic audio decoding to avoid optional torchcodec dependency
    hf_dataset = hf_dataset.cast_column(audio_column, HFAudio(decode=False))

    filesystem = getattr(hf_dataset, "_fs", None)
    
    print(f"✅ Loaded {len(hf_dataset)} samples from Hugging Face")
    print(f"📁 Dataset cached - future runs will use cached version!")
    
    # Limit samples if requested
    if max_samples and max_samples < len(hf_dataset):
        print(f"\n⚠️  Limiting to {max_samples} samples for testing")
        hf_dataset = hf_dataset.select(range(max_samples))
    
    # Prepare temp directory for audio files
    temp_audio_dir = output_dir / "temp_audio"
    temp_audio_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🎵 Processing samples...")
    dataset = []
    
    for idx, item in enumerate(tqdm(hf_dataset, desc="Processing")):
        try:
            # Get audio data (decoded=False keeps "array" field empty; use path + load)
            audio_data = item[audio_column]

            # Determine sampling rate
            sampling_rate = audio_data.get("sampling_rate", 16000)

            # Convert Hugging Face audio to numpy array if buffer available
            audio_array = None
            if "array" in audio_data and audio_data["array"] is not None:
                audio_array = audio_data["array"]
            elif "bytes" in audio_data and audio_data["bytes"] is not None:
                with sf.SoundFile(io.BytesIO(audio_data["bytes"])) as sf_desc:
                    audio_array = sf_desc.read(dtype="float32")
                    sampling_rate = sf_desc.samplerate
            elif "path" in audio_data and audio_data["path"]:
                audio_path = audio_data["path"]
                if filesystem and not os.path.isabs(audio_path):
                    with filesystem.open(audio_path, "rb") as f:
                        with sf.SoundFile(f) as sf_desc:
                            audio_array = sf_desc.read(dtype="float32")
                            sampling_rate = sf_desc.samplerate
                else:
                    audio_array, sampling_rate = librosa.load(
                        audio_path, sr=16000, mono=True
                    )
            else:
                audio_file = item.get("audio_file")
                print(f"Warning: No audio data available for sample {idx} ({audio_file})")
                continue

            if audio_array is None:
                audio_file = item.get("audio_file")
                print(f"Warning: Failed to decode audio for sample {idx} ({audio_file})")
                continue

            # Ensure numpy array
            if not isinstance(audio_array, np.ndarray):
                audio_array = np.asarray(audio_array)

            # Get duration
            duration = len(audio_array) / sampling_rate
            
            # Filter by duration
            if duration < min_duration or duration > max_duration:
                continue
            
            # Get transcription
            text = item.get(text_column, "")
            if not text:
                continue
            
            # Save audio to temp file
            audio_filename = temp_audio_dir / f"audio_{idx:06d}.wav"
            sf.write(str(audio_filename), audio_array, sampling_rate)
            
            # Add to dataset
            dataset.append({
                "audio_path": str(audio_filename.absolute()),
                "text": text,
                "language": language,
                "duration": duration,
            })
            
        except Exception as e:
            print(f"Warning: Error processing sample {idx}: {e}")
            continue
    
    print(f"\n✅ Processed {len(dataset)} valid samples")
    
    # Split into train/eval
    random.shuffle(dataset)
    
    # Use 90/10 split for HF datasets
    split_idx = int(len(dataset) * 0.9)
    train_dataset = dataset[:split_idx]
    eval_dataset = dataset[split_idx:]
    
    return train_dataset, eval_dataset


def main():
    """Main data preparation function."""
    args = parse_args()
    
    print("\n" + "="*70)
    print("📊 DATA PREPARATION FOR DUAL-ATTENTION WHISPER")
    print("="*70)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if using Hugging Face dataset or local files
    if args.hf_dataset:
        # Load from Hugging Face
        train_dataset, eval_dataset = prepare_hf_dataset(
            dataset_name=args.hf_dataset,
            config=args.hf_config,
            split=args.hf_split,
            audio_column=args.hf_audio_column,
            text_column=args.hf_text_column,
            language=args.language,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            cache_dir=args.cache_dir,
            max_samples=args.max_samples,
            output_dir=output_dir,
        )
    else:
        # Load from local files
        if not args.transcripts:
            raise ValueError("--transcripts is required when using --audio_dir")
        
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
