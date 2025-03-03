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

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class PCLDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(label, dtype=torch.long),
        }


class PCLClassifier(nn.Module):
    def __init__(self, model_name, num_classes=2):
        super(PCLClassifier, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)

        # Freeze all layers except the last 2
        for name, param in self.model.named_parameters():
            if "encoder.layer" in name:
                layer_num = int(name.split("encoder.layer.")[1].split(".")[0])
                total_layers = len(
                    [
                        n
                        for n, _ in self.model.named_parameters()
                        if "encoder.layer" in n and ".0." in n
                    ]
                )
                if layer_num < total_layers - 2:
                    param.requires_grad = False

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # Use CLS token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


def train_model(model, train_loader, val_loader, device, num_epochs=3):
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5
    )
    criterion = nn.CrossEntropyLoss(
        # weight=torch.tensor([1.0, 2.0]).to(device)
    )  # Higher weight for positive class

    best_f1 = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]"
        )

        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        val_loss = 0

        progress_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]")
        with torch.no_grad():
            for batch in progress_bar:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)

                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_val_loss = val_loss / len(val_loader)

        # Calculate metrics
        report = classification_report(val_labels, val_preds, output_dict=True)
        f1_pos = report["1"]["f1-score"]

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Validation F1 (Positive Class): {f1_pos:.4f}")
        print("-" * 50)

        # Save best model
        if f1_pos > best_f1:
            best_f1 = f1_pos
            best_model_state = model.state_dict().copy()
            print(f"New best model saved! F1: {f1_pos:.4f}")

    return best_model_state, best_f1


def predict_text(model, text, tokenizer, device, max_length=256):
    """
    Make a prediction for a single text input.

    Args:
        model: The trained PCLClassifier model
        text: String input text to classify
        tokenizer: The tokenizer used for the model
        device: The device to run inference on (cuda or cpu)
        max_length: Maximum token length (should match training)

    Returns:
        predicted_class: 0 or 1 representing the predicted class
        probabilities: Softmax probabilities for each class
    """
    # Ensure the model is in evaluation mode
    model.eval()

    # Tokenize the input text
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    # Move the inputs to the specified device
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    return predicted_class


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    # Load data
    print("Loading datasets...")
    train_df = pd.read_csv("data/train_data.csv")
    val_df = pd.read_csv("data/val_data.csv")
    print(f"Training data size: {len(train_df)}")

    # Initialize tokenizer and model
    print("Initializing model and tokenizer...")
    model_name = "roberta-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Prepare data
    train_dataset = PCLDataset(
        train_df["text"].values, train_df["label"].values, tokenizer
    )
    val_dataset = PCLDataset(val_df["text"].values, val_df["label"].values, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    # Prepare model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = PCLClassifier(model_name)
    model = model.to(device)

    if args.model is not None and Path(args.model).exists():
        best_model_state = torch.load(args.model, weights_only=True)
    else:
        # Train model
        print("\nStarting training...")
        best_model_state, best_f1 = train_model(model, train_loader, val_loader, device)
        print(f"\nTraining completed! Best validation F1: {best_f1:.4f}")

        # Save best model
        torch.save(best_model_state, "best_freezed.pt")
        print("Best model saved to best_freezed.pt")

    # Load best model for testing
    model.load_state_dict(best_model_state)
    model.eval()

    # Test evaluation
    print("\nEvaluating on test set...")

    test_df = pd.read_csv("data/test_data.csv")
    test_dataset = PCLDataset(
        test_df["text"].values, test_df["label"].values, tokenizer
    )
    test_loader = DataLoader(test_dataset, batch_size=16)

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

    # Print test results
    test_report = classification_report(test_labels, test_preds)
    print("\nTest Results:")
    print(test_report)


if __name__ == "__main__":
    main()
