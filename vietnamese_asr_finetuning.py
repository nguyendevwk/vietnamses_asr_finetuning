#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script Fine-tuning mô hình nhận dạng tiếng nói tiếng Việt dựa trên Wav2Vec2
Mô hình base: nguyenvulebinh/wav2vec2-base-vietnamese-250h
"""

import os
import re
import json
import random
import argparse
import numpy as np
import pandas as pd
import torch
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field

# Import Hugging Face libraries
from datasets import load_dataset, Audio, Dataset
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer
)
from evaluate import load as load_metric

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Fine-tune Vietnamese ASR model')

    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing audio files')
    parser.add_argument('--metadata_file', type=str, required=True,
                        help='Path to metadata CSV file with audio paths and transcriptions')
    parser.add_argument('--output_dir', type=str, default='./vietnamese-asr-finetuned',
                        help='Directory to save model checkpoints')

    # Training arguments
    parser.add_argument('--base_model', type=str,
                        default='nguyenvulebinh/wav2vec2-base-vietnamese-250h',
                        help='Base model to fine-tune')
    parser.add_argument('--max_train_samples', type=int, default=None,
                        help='Limit number of training samples (None for all)')
    parser.add_argument('--max_val_samples', type=int, default=None,
                        help='Limit number of validation samples (None for all)')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Validation split ratio')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Training batch size')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--evaluation_steps', type=int, default=400,
                        help='Evaluation frequency (in steps)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # Device arguments
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda, cpu, or None for auto-detection)')

    return parser.parse_args()

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_custom_dataset(data_dir, metadata_file):
    """
    Load dataset from a directory and metadata file.

    Args:
        data_dir: Directory containing audio files
        metadata_file: CSV file with 'audio_path' and 'transcription' columns

    Returns:
        Dataset object
    """
    print(f"Loading dataset from {metadata_file}...")

    try:
        # Load metadata
        df = pd.read_csv(metadata_file)

        # Make sure required columns exist
        required_cols = ['audio_path', 'transcription']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Metadata file must contain '{col}' column")

        # Create full paths
        df['audio_path'] = df['audio_path'].apply(lambda x: os.path.join(data_dir, x) if not os.path.isabs(x) else x)

        # Verify audio files exist
        missing_files = [path for path in df['audio_path'] if not os.path.exists(path)]
        if missing_files:
            print(f"Warning: {len(missing_files)} audio files not found. First few: {missing_files[:5]}")
            # Filter out missing files
            df = df[~df['audio_path'].isin(missing_files)]

        # Create dataset dictionary
        dataset_dict = {
            'audio_path': df['audio_path'].tolist(),
            'transcription': df['transcription'].tolist()
        }

        # Create Hugging Face dataset
        dataset = Dataset.from_dict(dataset_dict)

        # Add audio column
        dataset = dataset.cast_column("audio_path", Audio(sampling_rate=16000))

        # Rename audio_path to audio for compatibility with the rest of the code
        dataset = dataset.rename_column("audio_path", "audio")

        return dataset

    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

def show_random_elements(dataset, num_examples=5):
    """Display random examples from dataset."""
    assert num_examples <= len(dataset), "Can't pick more elements than available in the dataset."
    picks = random.sample(range(len(dataset)), num_examples)
    df = pd.DataFrame([dataset[i] for i in picks])
    return df[['transcription']]

def remove_special_characters(batch):
    """Clean transcriptions by removing special characters."""
    chars_to_remove_regex = r'[\,\?\.\!\-\;\:\"\"\%\'\"\�\']'
    batch["transcription"] = re.sub(chars_to_remove_regex, '', batch["transcription"]).lower()
    return batch

def extract_all_chars(batch):
    """Extract all unique characters from transcriptions."""
    all_text = " ".join(batch["transcription"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}

def prepare_dataset(batch, processor):
    """Prepare dataset for training."""
    audio = batch["audio"]

    # Extract input values
    batch["input_values"] = processor(
        audio["array"],
        sampling_rate=audio["sampling_rate"]
    ).input_values[0]

    # Calculate input length
    batch["input_length"] = len(batch["input_values"])

    # Extract labels
    with processor.as_target_processor():
        batch["labels"] = processor(batch["transcription"]).input_ids

    return batch

@dataclass
class DataCollatorCTCWithPadding:
    """Data collator that will dynamically pad the inputs received."""
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels since they have different lengths and need different padding
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # Pad inputs
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # Pad labels
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels

        return batch

class WerCallback(Trainer.EvalPrediction):
    """Custom callback to log WER after each evaluation."""
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None and "eval_wer" in metrics:
            wer = metrics["eval_wer"]
            step = state.global_step
            print(f"\n===== Evaluation at step {step} =====")
            print(f"Word Error Rate (WER): {wer:.4f}")

            # Transcribe a random sample for comparison
            test_idx = random.randint(0, len(self.eval_dataset)-1)
            test_audio = self.eval_dataset[test_idx]["input_values"]
            test_label = self.eval_dataset[test_idx]["labels"]

            # Decode reference label
            with self.processor.as_target_processor():
                test_label_str = self.processor.batch_decode([test_label], group_tokens=False)[0]

            # Get prediction
            inputs = torch.tensor([test_audio]).to(self.model.device)
            with torch.no_grad():
                logits = self.model(inputs).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            test_pred_str = self.processor.batch_decode(predicted_ids)[0]

            print(f"Sample transcription comparison:")
            print(f"Reference: {test_label_str}")
            print(f"Prediction: {test_pred_str}")
            print("====================================\n")

def compute_metrics(pred):
    """Compute Word Error Rate metric."""
    wer_metric = load_metric("wer")

    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    # Replace -100 with pad token ID
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    # Decode predictions and labels
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    # Compute WER
    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

def transcribe_sample(audio_path=None, audio_array=None, sampling_rate=16000, processor=None, model=None):
    """Transcribe an audio file or array using the fine-tuned model."""
    if audio_path:
        # Load audio from file
        import librosa
        audio_input, sample_rate = librosa.load(audio_path, sr=sampling_rate)
    elif audio_array is not None:
        # Use provided audio array
        audio_input = audio_array
    else:
        raise ValueError("Either audio_path or audio_array must be provided")

    # Preprocess audio
    inputs = processor(audio_input, sampling_rate=sampling_rate, return_tensors="pt", padding=True)

    # Move inputs to appropriate device
    device = model.device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get logits
    with torch.no_grad():
        logits = model(**inputs).logits

    # Get predicted IDs
    predicted_ids = torch.argmax(logits, dim=-1)

    # Decode
    transcription = processor.batch_decode(predicted_ids)

    return transcription[0]

def main():
    # Parse arguments
    args = parse_arguments()

    # Set random seed
    set_seed(args.seed)

    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    dataset = load_custom_dataset(args.data_dir, args.metadata_file)

    # Split dataset
    train_val_split = dataset.train_test_split(test_size=args.val_split, seed=args.seed)
    train_dataset = train_val_split["train"]
    val_dataset = train_val_split["test"]

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Show some examples
    print("\nSample transcriptions:")
    print(show_random_elements(train_dataset))

    # Clean transcriptions
    train_dataset = train_dataset.map(remove_special_characters)
    val_dataset = val_dataset.map(remove_special_characters)

    # Extract vocabulary
    vocab_train = train_dataset.map(
        extract_all_chars,
        batched=True,
        batch_size=-1,
        keep_in_memory=False,
        remove_columns=train_dataset.column_names
    )

    vocab_val = val_dataset.map(
        extract_all_chars,
        batched=True,
        batch_size=-1,
        keep_in_memory=False,
        remove_columns=val_dataset.column_names
    )

    # Combine vocabularies
    vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_val["vocab"][0]))

    print(f"\nVocabulary size: {len(vocab_list)}")
    print(f"Sample characters: {vocab_list[:20]}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load processor directly from base model to maintain compatibility
    print(f"\nLoading base model and processor from {args.base_model}")
    processor = Wav2Vec2Processor.from_pretrained(args.base_model)

    # Prepare datasets
    print("\nPreparing datasets for training...")

    # Limit dataset size if specified
    if args.max_train_samples:
        train_dataset = train_dataset.select(range(min(args.max_train_samples, len(train_dataset))))

    if args.max_val_samples:
        val_dataset = val_dataset.select(range(min(args.max_val_samples, len(val_dataset))))

    # Apply preprocessing
    prepare_fn = lambda batch: prepare_dataset(batch, processor)

    train_dataset = train_dataset.map(
        prepare_fn,
        remove_columns=train_dataset.column_names,
        num_proc=4
    )

    val_dataset = val_dataset.map(
        prepare_fn,
        remove_columns=val_dataset.column_names,
        num_proc=4
    )

    print(f"Processed training samples: {len(train_dataset)}")
    print(f"Processed validation samples: {len(val_dataset)}")

    # Create data collator
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    # Initialize model
    print("\nInitializing model...")
    model = Wav2Vec2ForCTC.from_pretrained(
        args.base_model,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        layerdrop=0.0,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        ignore_mismatched_sizes=True
    )

    # Freeze feature extractor
    model.freeze_feature_extractor()
    model = model.to(device)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        group_by_length=True,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        evaluation_strategy="steps",
        num_train_epochs=args.epochs,
        gradient_checkpointing=True,
        fp16=True,
        save_steps=args.evaluation_steps,
        eval_steps=args.evaluation_steps,
        logging_steps=args.evaluation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=500,
        save_total_limit=2,
        push_to_hub=False,
        remove_unused_columns=False,
        report_to="tensorboard",
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
    )

    # Create WER callback for detailed evaluation reporting
    wer_callback = WerCallback()
    wer_callback.processor = processor
    wer_callback.eval_dataset = val_dataset
    wer_callback.model = model

    # Initialize trainer
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=processor,
        callbacks=[wer_callback]
    )

    # Start training
    print("\nStarting training...")
    trainer.train()

    # Save model
    print("\nSaving model...")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    # Final evaluation
    print("\nPerforming final evaluation...")
    eval_results = trainer.evaluate()
    print(f"Final evaluation results: {eval_results}")

    # Test on a random sample
    print("\nTranscribing a test sample:")
    test_idx = random.randint(0, len(val_dataset)-1)
    test_audio = val_dataset[test_idx]["input_values"]
    test_label = val_dataset[test_idx]["labels"]

    # Decode reference label
    with processor.as_target_processor():
        test_label_str = processor.batch_decode([test_label], group_tokens=False)[0]

    # Transcribe test audio
    test_pred_str = transcribe_sample(audio_array=test_audio, processor=processor, model=model)

    print(f"Reference: {test_label_str}")
    print(f"Prediction: {test_pred_str}")
    wer_metric = load_metric("wer")
    print(f"WER: {wer_metric.compute(predictions=[test_pred_str], references=[test_label_str])}")

    print("\nTraining completed successfully!")

if __name__ == "__main__":
    # Install required packages if needed
    try:
        import datasets
        import evaluate
        import jiwer
    except ImportError:
        print("Installing required packages...")
        import subprocess
        subprocess.check_call(["pip", "install", "-q", "transformers", "datasets", "evaluate", "jiwer", "fsspec==2023.6.0"])

    main()