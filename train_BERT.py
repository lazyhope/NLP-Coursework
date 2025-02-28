import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    IntervalStrategy,
)

# File paths to your CSV data
TRAIN_PATH = "data/train_data.csv"
VAL_PATH = "data/val_data.csv"
TEST_PATH = "data/test_data.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define metric function for evaluation (for the Trainer)
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# Custom dataset class for BERT
class PatronizingDataset(Dataset):
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
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }

def load_data():
    print("Loading data...")
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train, y_train = train_df["text"].tolist(), train_df["label"].tolist()
    X_val, y_val = val_df["text"].tolist(), val_df["label"].tolist()
    X_test, y_test = test_df["text"].tolist(), test_df["label"].tolist()

    print(f"Training examples: {len(X_train)}")
    print(f"Validation examples: {len(X_val)}")
    print(f"Test examples: {len(X_test)}")

    pos = sum(y_train)
    neg = len(y_train) - pos
    print(f"Class distribution - Positive: {pos}, Negative: {neg}, Ratio: {pos / neg:.3f}")

    return X_train, y_train, X_val, y_val, X_test, y_test

def main():
    # Load tokenizer and model (BERT)
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    # Create dataset objects
    train_dataset = PatronizingDataset(X_train, y_train, tokenizer)
    val_dataset = PatronizingDataset(X_val, y_val, tokenizer)
    test_dataset = PatronizingDataset(X_test, y_test, tokenizer)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=10,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        logging_dir="./logs",
        learning_rate=2e-5,
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=100,
        logging_steps=10,
        load_best_model_at_end=False,         # Not using early stopping, so no best model tracking
        fp16=torch.cuda.is_available(),       # Mixed precision if GPU is available
        report_to=["tensorboard"],            # Logging to TensorBoard (can be set to [] if not needed)
    )


    # Initialize Trainer with early stopping
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Train the model
    print("Starting training...")
    trainer.train()

    # Evaluate on validation set (using Trainer's evaluate)
    print("\n=== Validation Performance ===")
    val_results = trainer.evaluate(val_dataset)
    print("Validation results:", val_results)

    # Predict on test set
    print("\n=== Test Performance ===")
    test_output = trainer.predict(test_dataset)
    test_preds = test_output.predictions.argmax(-1)

    # Print detailed classification report for test data
    print(classification_report(y_test, test_preds, digits=3))

    # Compute overall metrics on test data
    test_metrics = compute_metrics(test_output)
    print("Overall test metrics:", test_metrics)

    # Save the model and tokenizer
    model_path = "./model_final"
    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
