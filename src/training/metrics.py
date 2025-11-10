"""
Evaluation metrics for ASR.

Includes:
- Word Error Rate (WER)
- Character Error Rate (CER)
"""

import evaluate
from typing import Dict
import numpy as np


# Load metrics
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")


def compute_metrics(processor):
    """
    Create a metrics computation function for Seq2SeqTrainer.
    
    Args:
        processor: WhisperProcessor instance for decoding predictions
        
    Returns:
        Function that computes WER and CER
    """
    
    def _compute_metrics(pred) -> Dict[str, float]:
        """
        Compute WER and CER for predictions.
        
        Args:
            pred: EvalPrediction object with predictions and label_ids
            
        Returns:
            Dictionary with "wer" and "cer" metrics
        """
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        
        # Replace -100 with pad_token_id (for decoding)
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        
        # Decode predictions and references
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
        
        # Compute WER
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        
        # Compute CER
        cer = cer_metric.compute(predictions=pred_str, references=label_str)
        
        return {
            "wer": wer * 100,  # Convert to percentage
            "cer": cer * 100,
        }
    
    return _compute_metrics


def evaluate_model(model, processor, test_dataset, device="cuda"):
    """
    Evaluate model on test dataset and print detailed results.
    
    Args:
        model: Trained model
        processor: WhisperProcessor
        test_dataset: Test dataset
        device: Device to run evaluation on
        
    Returns:
        Dictionary with metrics and sample predictions
    """
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    
    model.eval()
    model.to(device)
    
    # Create dataloader
    dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    all_predictions = []
    all_references = []
    
    print("\n🔍 Evaluating model...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # Move to device
            input_features = batch["input_features"].to(device)
            labels = batch["labels"].to(device)
            
            # Generate predictions
            generated_ids = model.generate(
                input_features,
                max_length=448,
                num_beams=5,
                language="az",  # Change as needed
                task="transcribe",
            )
            
            # Decode
            predictions = processor.batch_decode(generated_ids, skip_special_tokens=True)
            references = processor.batch_decode(labels, skip_special_tokens=True)
            
            all_predictions.extend(predictions)
            all_references.extend(references)
    
    # Compute metrics
    wer = wer_metric.compute(predictions=all_predictions, references=all_references)
    cer = cer_metric.compute(predictions=all_predictions, references=all_references)
    
    print("\n" + "="*60)
    print("📊 EVALUATION RESULTS")
    print("="*60)
    print(f"WER: {wer * 100:.2f}%")
    print(f"CER: {cer * 100:.2f}%")
    print("="*60)
    
    # Show sample predictions
    print("\n📝 Sample Predictions:")
    for i in range(min(5, len(all_predictions))):
        print(f"\n[{i+1}]")
        print(f"  Reference:  {all_references[i]}")
        print(f"  Prediction: {all_predictions[i]}")
    
    return {
        "wer": wer * 100,
        "cer": cer * 100,
        "predictions": all_predictions,
        "references": all_references,
    }
