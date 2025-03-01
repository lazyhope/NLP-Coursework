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

# ------------------------------
# 1. File Paths
# ------------------------------
TRAIN_PATH = "data/train_data.csv"
VAL_PATH   = "data/val_data.csv"
TEST_PATH  = "data/test_data.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------------
# 2. Metric Function
# ------------------------------
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# ------------------------------
# 3. Custom Dataset Class for BERT
# ------------------------------
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

# ------------------------------
# 4. Load Data Function
# ------------------------------
def load_data():
    print("Loading data...")
    train_df = pd.read_csv(TRAIN_PATH)
    val_df   = pd.read_csv(VAL_PATH)
    test_df  = pd.read_csv(TEST_PATH)

    X_train, y_train = train_df["text"].tolist(), train_df["label"].tolist()
    X_val,   y_val   = val_df["text"].tolist(),   val_df["label"].tolist()
    X_test,  y_test  = test_df["text"].tolist(),  test_df["label"].tolist()

    print(f"Training examples: {len(X_train)}")
    print(f"Validation examples: {len(X_val)}")
    print(f"Test examples: {len(X_test)}")

    pos = sum(y_train)
    neg = len(y_train) - pos
    print(f"Class distribution - Positive: {pos}, Negative: {neg}, Ratio: {pos / (neg+1e-6):.3f}")

    return X_train, y_train, X_val, y_val, X_test, y_test

# ------------------------------
# 5. Custom Callback to Log Training & Validation F1
# ------------------------------
class F1LoggerCallback(TrainerCallback):
    def __init__(self, train_dataset):
        self.train_dataset = train_dataset
        self.trainer = None  # to be set manually later
        self.eval_steps = []
        self.val_f1_scores = []
        self.train_f1_scores = []

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # Log validation F1 (assumes compute_metrics returns "eval_f1")
        if metrics and "eval_f1" in metrics:
            self.eval_steps.append(state.global_step)
            self.val_f1_scores.append(metrics["eval_f1"])
        # Use the attached trainer to compute training F1
        train_output = self.trainer.predict(self.train_dataset)
        train_preds = train_output.predictions.argmax(-1)
        train_labels = train_output.label_ids
        # Create an object mimicking the expected structure for compute_metrics
        train_metrics = compute_metrics(
            type("obj", (object,), {"label_ids": train_labels, "predictions": train_output.predictions})
        )
        self.train_f1_scores.append(train_metrics["f1"])
        return control

# ------------------------------
# 6. Plot F1 Learning Curve Function
# ------------------------------
def plot_f1_learning_curve(steps, train_f1, val_f1, filename="BERT_f1_learning_curve.png"):
    plt.figure(figsize=(8, 6))
    plt.plot(steps, train_f1, marker="o", label="Training F1 Score")
    plt.plot(steps, val_f1, marker="o", label="Validation F1 Score")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title("Learning Curve (F1 Score) of BERT")
    plt.ylim(0.0, 1.0)
    plt.grid(True)
    plt.legend()
    plt.savefig(filename)
    plt.show()

# ------------------------------
# 7. Main Function
# ------------------------------
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
        load_best_model_at_end=False,
        fp16=torch.cuda.is_available(),
        report_to=["tensorboard"],
    )

    # Initialize our custom F1 callback with the training dataset
    f1_callback = F1LoggerCallback(train_dataset)

    # Initialize Trainer without callbacks first
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    # Manually attach the trainer to our callback and add the callback to trainer
    f1_callback.trainer = trainer
    trainer.add_callback(f1_callback)

    # Train the model (this trains only once, logging eval metrics periodically)
    print("Starting training...")
    trainer.train()

    # Final evaluation on validation set
    print("\n=== Final Validation Performance ===")
    val_results = trainer.evaluate(val_dataset)
    print("Validation results:", val_results)
    val_preds = trainer.predict(val_dataset).predictions.argmax(-1)
    print(classification_report(y_val, val_preds, digits=3))

    # Evaluate on test set
    print("\n=== Test Performance ===")
    test_output = trainer.predict(test_dataset)
    test_preds = test_output.predictions.argmax(-1)
    print(classification_report(y_test, test_preds, digits=3))
    test_metrics = compute_metrics(test_output)
    print("Overall test metrics:", test_metrics)

    # Save the model and tokenizer
    model_path = "./model_final"
    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"Model saved to {model_path}")

    # Plot the F1 learning curve using data logged by our callback
    print("\n=== Plotting F1 Learning Curve ===")
    plot_f1_learning_curve(f1_callback.eval_steps, f1_callback.train_f1_scores, f1_callback.val_f1_scores,
                             filename="BERT_f1_learning_curve.png")

if __name__ == "__main__":
    main()
