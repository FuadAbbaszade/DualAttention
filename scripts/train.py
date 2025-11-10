#!/usr/bin/env python3
"""
Training script for Dual-Attention Whisper.

This script uses the optimized training configuration for maximum GPU performance.
"""

import os
import sys
import argparse
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformers import (
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

from model.dual_whisper import create_dual_attention_whisper
from data.dataset import NoisyAudioDataset
from data.collator import DataCollatorSpeechSeq2SeqWithPadding
from training.metrics import compute_metrics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Dual-Attention Whisper")
    
    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="openai/whisper-small",
        help="Pre-trained Whisper model to use"
    )
    parser.add_argument(
        "--freeze_encoder",
        action="store_true",
        default=True,
        help="Freeze encoder during training"
    )
    parser.add_argument(
        "--freeze_primary_decoder",
        action="store_true",
        default=False,
        help="Freeze primary decoder attention"
    )
    
    # Data arguments
    parser.add_argument(
        "--train_data",
        type=str,
        required=True,
        help="Path to training data JSON"
    )
    parser.add_argument(
        "--eval_data",
        type=str,
        required=True,
        help="Path to evaluation data JSON"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="az",
        help="Language code (e.g., 'az' for Azerbaijani)"
    )
    
    # Training arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=None,
        help="Number of training epochs (alternative to max_steps)"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=10000,
        help="Maximum training steps"
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size per GPU for training"
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=32,
        help="Batch size per GPU for evaluation"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Learning rate"
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=1000,
        help="Warmup steps"
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=1000,
        help="Evaluation frequency"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1000,
        help="Checkpoint saving frequency"
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Logging frequency"
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    print("\n" + "="*70)
    print("🚀 DUAL-ATTENTION WHISPER TRAINING")
    print("="*70)
    
    # ========================================================================
    # 1. LOAD MODEL AND PROCESSOR
    # ========================================================================
    print("\n📦 Loading model and processor...")
    
    processor = WhisperProcessor.from_pretrained(
        args.model_name,
        language=args.language,
        task="transcribe"
    )
    
    model = create_dual_attention_whisper(
        model_name=args.model_name,
        freeze_encoder=args.freeze_encoder,
        freeze_primary_decoder=args.freeze_primary_decoder,
    )
    
    # ========================================================================
    # 2. PREPARE DATASETS
    # ========================================================================
    print("\n📊 Loading datasets...")
    
    train_dataset = NoisyAudioDataset(
        data_path=args.train_data,
        processor=processor,
        language=args.language,
    )
    
    eval_dataset = NoisyAudioDataset(
        data_path=args.eval_data,
        processor=processor,
        language=args.language,
    )
    
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Eval samples:  {len(eval_dataset)}")
    
    # ========================================================================
    # 3. DATA COLLATOR
    # ========================================================================
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )
    
    # ========================================================================
    # 4. TRAINING ARGUMENTS - MAXIMUM PERFORMANCE
    # ========================================================================
    print("\n⚙️  Configuring training arguments...")
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        
        # Batch settings - GPU'ları zorla
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=1,
        
        # Learning rate
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps if args.num_train_epochs is None else -1,
        num_train_epochs=args.num_train_epochs if args.num_train_epochs else 3,
        weight_decay=0.01,
        
        # Evaluation & Saving
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=5,
        
        # Generation
        predict_with_generate=True,
        generation_max_length=100,
        generation_num_beams=1,  # Speed optimization during training eval
        
        # Logging
        logging_steps=args.logging_steps,
        logging_first_step=True,
        report_to="tensorboard",
        
        # Model selection
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        
        # Performance optimizations
        fp16=torch.cuda.is_available(),
        fp16_full_eval=torch.cuda.is_available(),
        dataloader_num_workers=16,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        
        # Multi-GPU
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=25,
        
        # Memory & Speed
        gradient_checkpointing=False,
        optim="adamw_torch_fused",
        
        # Misc
        remove_unused_columns=False,
        push_to_hub=False,
        disable_tqdm=False,
        
        # Advanced
        tf32=True if torch.cuda.is_available() else False,
        auto_find_batch_size=False,
    )
    
    # ========================================================================
    # 5. METRICS
    # ========================================================================
    metrics_fn = compute_metrics(processor)
    
    # ========================================================================
    # 6. TRAINER
    # ========================================================================
    print("\n🏋️  Initializing trainer...")
    
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=metrics_fn,
        processing_class=processor.feature_extractor,
    )
    
    # ========================================================================
    # 7. PRINT CONFIGURATION
    # ========================================================================
    print("\n" + "="*70)
    print("📋 TRAINING CONFIGURATION")
    print("="*70)
    print(f"Model: {args.model_name}")
    print(f"Language: {args.language}")
    print(f"Output directory: {output_dir}")
    print(f"\nGPU Information:")
    if torch.cuda.is_available():
        print(f"  GPU count: {torch.cuda.device_count()}")
        print(f"  GPU model: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
    else:
        print("  Running on CPU (WARNING: Very slow!)")
    print(f"\nBatch Configuration:")
    print(f"  Per-device train batch: {args.per_device_train_batch_size}")
    print(f"  Per-device eval batch: {args.per_device_eval_batch_size}")
    print(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
    if torch.cuda.is_available():
        effective_batch = (
            args.per_device_train_batch_size *
            training_args.gradient_accumulation_steps *
            torch.cuda.device_count()
        )
        print(f"  Effective batch size: {effective_batch}")
    print(f"\nOptimization:")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Warmup steps: {args.warmup_steps}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Optimizer: {training_args.optim}")
    print(f"  FP16: {training_args.fp16}")
    print(f"  TF32: {training_args.tf32}")
    print(f"\nData:")
    print(f"  Dataloader workers: {training_args.dataloader_num_workers}")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Eval samples: {len(eval_dataset)}")
    print("="*70 + "\n")
    
    # ========================================================================
    # 8. TRAIN!
    # ========================================================================
    print("🎯 Starting training...\n")
    
    try:
        train_result = trainer.train()
        
        # Save final model
        print("\n💾 Saving final model...")
        trainer.save_model(output_dir)
        processor.save_pretrained(output_dir)
        
        # Print final metrics
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETED!")
        print("="*70)
        print(f"📁 Model saved to: {output_dir}")
        print(f"🎯 Final metrics:")
        for key, value in train_result.metrics.items():
            print(f"   {key}: {value}")
        print("="*70 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print("💾 Saving current checkpoint...")
        trainer.save_model(output_dir)
        processor.save_pretrained(output_dir)
        print(f"📁 Checkpoint saved to: {output_dir}")
        
    except Exception as e:
        print(f"\n\n❌ Training failed with error:")
        print(f"   {str(e)}")
        raise


if __name__ == "__main__":
    main()
