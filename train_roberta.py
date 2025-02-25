import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import Dataset
from transformers import (
    EarlyStoppingCallback,
    IntervalStrategy,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    Trainer,
    TrainingArguments,
)

TRAIN_PATH = "data/train_data.csv"
VAL_PATH = "data/val_data.csv"
TEST_PATH = "data/test_data.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Define metric function for evaluation
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


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
            return_token_type_ids=False,
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

    # Extract features and labels
    X_train, y_train = train_df["text"].tolist(), train_df["label"].tolist()
    X_val, y_val = val_df["text"].tolist(), val_df["label"].tolist()
    X_test, y_test = test_df["text"].tolist(), test_df["label"].tolist()


    print(f"Training examples: {len(X_train)}")
    print(f"Validation examples: {len(X_val)}")
    print(f"Test examples: {len(X_test)}")

    # Count class distribution
    orig_pos = sum(y_train)
    orig_neg = len(y_train) - orig_pos

    print(
        f"Original class distribution - Positive: {orig_pos}, Negative: {orig_neg}, Ratio: {orig_pos / orig_neg:.3f}"
    )

    return X_train, y_train, X_val, y_val, X_test, y_test


def main():
    # Load tokenizer and model
    model_name = "roberta-large"
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(
        model_name, num_labels=2, problem_type="single_label_classification"
    )

    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    # Create datasets
    train_dataset = PatronizingDataset(X_train, y_train, tokenizer)
    val_dataset = PatronizingDataset(X_val, y_val, tokenizer)
    test_dataset = PatronizingDataset(X_test, y_test, tokenizer)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=10,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy=IntervalStrategy.STEPS,
        eval_steps=100,
        save_strategy=IntervalStrategy.STEPS,
        save_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,  # Only keep the 2 best checkpoints
        fp16=True,  # Mixed precision training for faster performance
        gradient_accumulation_steps=1,  # Can increase if needed for larger effective batch size
        learning_rate=2e-5,
        report_to=["tensorboard"],  # Only use tensorboard, disable wandb
    )

    # Initialize Trainer with early stopping
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # Train the model
    print("Starting training...")
    trainer.train()

    # Evaluate on test set
    print("Evaluating on test set...")
    test_results = trainer.evaluate(test_dataset)
    print(f"Test Results: {test_results}")

    # Save the model
    model_path = "./model_final"
    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
