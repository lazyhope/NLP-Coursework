import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from utils import copy_upsampling, downsampling, smote_upsampling, adasyn_upsampling, preprocess_dataframe, apply_synonym_replacement, apply_back_translation, add_keyword_prefix
from train import PCLDataset, PCLClassifier, train_model, predict_text

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="Path to a pre-trained model file.")
    # Add boolean flags for sampling methods
    parser.add_argument("--preprocess", action="store_true", help="Apply preprocessing to train data.")
    parser.add_argument("--copy_upsampling", action="store_true", help="Apply copy upsampling to train data.")
    parser.add_argument("--downsampling", action="store_true", help="Apply downsampling to train data.")
    parser.add_argument("--smote_upsampling", action="store_true", help="Apply SMOTE upsampling to train data.")
    parser.add_argument("--adasyn_upsampling", action="store_true", help="Apply ADASYN upsampling to train data.")
    parser.add_argument("--synonym_augmentation", action="store_true", help="Apply synonym augmentation to train data.")
    parser.add_argument("--back_translation", action="store_true", help="Apply back-translation augmentation to train data.")
    parser.add_argument("--add_keyword_prefix", action="store_true", help="Apply keyword prefix augmentation to train data.")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training.")
    args = parser.parse_args()

    # Load training and validation data
    print("Loading datasets...")
    train_df = pd.read_csv("data/train_data_bt.csv")
    val_df = pd.read_csv("data/val_data.csv")
    print(f"Training data size: {len(train_df)}")

    # Apply preprocessing if specified
    if args.preprocess:
        print("Applying preprocessing to training data...")
        train_df = preprocess_dataframe(train_df)
        print("Preprocessing completed.")
    else:
        print("No preprocessing applied.")

    # Determine sampling method based on command-line flags (only one method applied at a time)
    sampling_func = None
    if args.copy_upsampling:
        print("Using copy upsampling.")
        sampling_func = copy_upsampling
    elif args.downsampling:
        print("Using downsampling.")
        sampling_func = downsampling
    elif args.smote_upsampling:
        print("Using SMOTE upsampling.")
        sampling_func = smote_upsampling
    elif args.adasyn_upsampling:
        print("Using ADASYN upsampling.")
        sampling_func = adasyn_upsampling
    else:
        print("No sampling method applied.")
    
    augmentation_func = None
    if args.synonym_augmentation:
        print("Using synonym augmentation.")
        augmentation_func = apply_synonym_replacement
    elif args.back_translation:
        print("Using back-translation augmentation.")
        augmentation_func = apply_back_translation
    elif args.add_keyword_prefix:
        print("Using keyword prefix augmentation.")
        augmentation_func = add_keyword_prefix
    else:
        print("No augmentation method applied.")
        

    # if augmentation_func is not None:
    #     train_df = augmentation_func(train_df)
    # Only process the training set with sampling (validation set remains untouched)
    if sampling_func is not None:
        train_df = sampling_func(train_df)
    
    if augmentation_func is not None:
        train_df = augmentation_func(train_df)

    # Initialize tokenizer and model
    print("Initializing model and tokenizer...")
    model_name = "roberta-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Prepare datasets and dataloaders
    train_dataset = PCLDataset(train_df["text"].values, train_df["label"].values, tokenizer)
    val_dataset = PCLDataset(val_df["text"].values, val_df["label"].values, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Prepare model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = PCLClassifier(model_name)
    model = model.to(device)

    # Load pre-trained model if provided, otherwise train
    if args.model is not None and Path(args.model).exists():
        best_model_state = torch.load(args.model, weights_only=True)
    else:
        print("\nStarting training...")
        best_model_state, best_f1 = train_model(model, train_loader, val_loader, device, num_epochs=args.num_epochs)
        print(f"\nTraining completed! Best validation F1: {best_f1:.4f}")
        torch.save(best_model_state, "best_freezed.pt")
        print("Best model saved to best_freezed.pt")

    # Load best model for testing
    model.load_state_dict(best_model_state)
    model.eval()

    # Test evaluation
    print("\nEvaluating on test set...")
    test_df = pd.read_csv("data/test_data.csv")
    test_dataset = PCLDataset(test_df["text"].values, test_df["label"].values, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    test_preds = []
    test_labels = []

    progress_bar = tqdm(test_loader, desc="Testing")
    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs, dim=1)

            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    test_report = classification_report(test_labels, test_preds)
    print("\nTest Results:")
    print(test_report)

    # Save the test report to a text file for later comparison
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    if sampling_func is not None:
        sampling_name = sampling_func.__name__
    else:
        sampling_name = "no_sampling"
    if args.preprocess:
        sampling_name = f"{sampling_name}_preprocessed"
    if augmentation_func is not None:
        augmentation_func_name = augmentation_func.__name__
        sampling_name = f"{sampling_name}_{augmentation_func_name}"
    report_file = results_dir / f"{sampling_name}_report.txt"
    with open(report_file, "w") as f:
        f.write(test_report)
    print(f"Test report saved to {report_file}")


if __name__ == "__main__":
    main()