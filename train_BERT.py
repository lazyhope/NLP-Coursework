import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
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
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
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
    print(f"Class distribution - Positive: {pos}, Negative: {neg}, Ratio: {pos / (neg+1e-6):.3f}")

    return X_train, y_train, X_val, y_val, X_test, y_test

# Custom callback to log F1 scores during evaluation
class F1LoggerCallback(TrainerCallback):
    def __init__(self):
        self.eval_steps = []
        self.f1_scores = []

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # metrics keys are prefixed with "eval_"
        if metrics and "eval_f1" in metrics:
            self.eval_steps.append(state.global_step)
            self.f1_scores.append(metrics["eval_f1"])
        return control

# Function to plot F1 learning curve
def plot_f1_learning_curve(steps, f1_scores, filename="BERT_f1_learning_curve.png"):
    plt.figure(figsize=(8, 6))
    plt.plot(steps, f1_scores, marker="o", label="Validation F1 Score")
    plt.xlabel("Global Step")
    plt.ylabel("F1 Score")
    plt.title("Learning Curve (F1 Score)")
    plt.ylim(0.0, 1.0)
    plt.grid(True)
    plt.legend()
    plt.savefig(filename)
    plt.show()

def main():
    # Load tokenizer and model (BERT)
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    # Shuffle the training data
    X_train, y_train = shuffle(X_train, y_train, random_state=42)

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
        evaluation_strategy="steps",  # Evaluate every fixed number of steps
        eval_steps=100,
        save_steps=100,
        logging_steps=10,
        load_best_model_at_end=False,  # Not using early stopping here
        fp16=torch.cuda.is_available(),  # Mixed precision if GPU is available
        report_to=["tensorboard"],
    )

    # Initialize our custom F1 callback
    f1_callback = F1LoggerCallback()

    # Initialize Trainer with the callback
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[f1_callback],
    )

    # Train the model
    print("Starting training...")
    trainer.train()

    # Evaluate on validation set
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

    # Plot the F1 learning curve (using the data logged by our callback)
    print("\n=== Plotting F1 Learning Curve ===")
    plot_f1_learning_curve(f1_callback.eval_steps, f1_callback.f1_scores, filename="f1_learning_curve.png")

if __name__ == "__main__":
    main()
