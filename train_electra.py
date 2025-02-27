from transformers import ElectraForSequenceClassification, ElectraTokenizer, TrainingArguments, Trainer, IntervalStrategy
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch.utils.data import Dataset

TRAIN_PATH = "data/train_data.csv"
VAL_PATH = "data/val_data.csv"
TEST_PATH = "data/test_data.csv"
MODEL_PATH = "model/electra_model_final"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# PatronizingDataset class
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

# Helper functions
def compute_metrics(pred):
    logits, labels = pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

def upsample(data):
    pos = data[data["label"] == 1]
    neg = data[data["label"] == 0]
    pos = pd.concat([pos]*10, ignore_index=True)
    
    return pd.concat([pos, neg])

def load_data(upsample=False):
    print("Loading data...")
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)

    if upsample:
        train_df = upsample(train_df)
        val_df = upsample(val_df)

    X_train, y_train = train_df["text"], train_df["label"]
    X_val, y_val = val_df["text"], val_df["label"]
    X_test, y_test = test_df["text"], test_df["label"]

    print(f"Train size: {len(X_train)}")
    print(f"Val size: {len(X_val)}")
    print(f"Test size: {len(X_test)}")

    return X_train, y_train, X_val, y_val, X_test, y_test

# Main function
def main():
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model_name = "google/electra-large-discriminator"
    tokenizer = ElectraTokenizer.from_pretrained(model_name)
    model = ElectraForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(upsample=False)

    # Create datasets
    train_dataset = PatronizingDataset(X_train, y_train, tokenizer)
    val_dataset = PatronizingDataset(X_val, y_val, tokenizer)
    test_dataset = PatronizingDataset(X_test, y_test, tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results/electra_results",
        evaluation_strategy=IntervalStrategy.STEPS,
        eval_steps=100,
        save_strategy=IntervalStrategy.STEPS,
        save_steps=100,
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=10,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        gradient_accumulation_steps=4,
        fp16=True,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print("Training the model...")
    trainer.train()

    print("Evaluating the model...")
    trainer.evaluate()

    print("Predicting on the test set...")
    predictions = trainer.predict(test_dataset)
    print(predictions.metrics)

    # Save the model
    trainer.save_model(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
