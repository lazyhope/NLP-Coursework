import argparse
import json

import optuna
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
    pos = sum(y_train)
    neg = len(y_train) - pos

    print(
        f"Original class distribution - Positive: {pos}, Negative: {neg}, Ratio: {pos / neg:.3f}"
    )

    return X_train, y_train, X_val, y_val, X_test, y_test


def model_init():
    """Function that initializes a new model for each hyperparameter search trial."""
    model_name = "roberta-large"
    return RobertaForSequenceClassification.from_pretrained(
        model_name, num_labels=2, problem_type="single_label_classification"
    )


def optuna_hp_space(trial: optuna.Trial):
    """Define the hyperparameter search space for Optuna."""
    return {
        "learning_rate": trial.suggest_categorical(
            "learning_rate", [1e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4]
        ),
        "per_device_train_batch_size": trial.suggest_categorical(
            "per_device_train_batch_size", [8, 16, 32, 64]
        ),
        "weight_decay": trial.suggest_categorical(
            "weight_decay", [0.0, 0.01, 0.05, 0.1, 0.15, 0.2]
        ),
        "warmup_ratio": trial.suggest_categorical(
            "warmup_ratio", [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
        ),
        "num_train_epochs": trial.suggest_categorical(
            "num_train_epochs", [2, 3, 5, 7, 10, 15, 20]
        ),
        "seed": trial.suggest_categorical("seed", [0, 21, 42]),
        "gradient_accumulation_steps": trial.suggest_categorical(
            "gradient_accumulation_steps", [1, 2, 4]
        ),
        "lr_scheduler_type": trial.suggest_categorical(
            "lr_scheduler_type", ["linear", "cosine", "cosine_with_restarts"]
        ),
    }


def compute_objective(metrics):
    """
    Compute the objective to be maximized/minimized during hyperparameter search.
    In this case, we want to maximize F1 score.
    """
    return metrics["eval_f1"]


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train a classifier with hyperparameter search"
    )
    parser.add_argument(
        "--do_hpo",
        action="store_true",
        help="Whether to perform hyperparameter optimization",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=20,
        help="Number of trials for hyperparameter search",
    )
    parser.add_argument(
        "--use_best_params",
        action="store_true",
        help="Use the best parameters from a previous HPO run",
    )
    args = parser.parse_args()

    # Load tokenizer
    model_name = "roberta-large"
    tokenizer = RobertaTokenizer.from_pretrained(model_name)

    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    # Create datasets
    train_dataset = PatronizingDataset(X_train, y_train, tokenizer)
    val_dataset = PatronizingDataset(X_val, y_val, tokenizer)
    test_dataset = PatronizingDataset(X_test, y_test, tokenizer)

    # Define baseline training arguments (these will be overridden by HPO if enabled)
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        warmup_ratio=0.1,
        seed=42,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        eval_strategy=IntervalStrategy.STEPS,
        eval_steps=100,
        save_strategy=IntervalStrategy.STEPS,
        save_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        fp16=True,
        gradient_accumulation_steps=1,
        learning_rate=2e-5,
        report_to=["tensorboard"],
    )

    if args.do_hpo:
        print("Performing hyperparameter optimization...")

        # Initialize Trainer without a model (model_init will be used)
        trainer = Trainer(
            model=None,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
            model_init=model_init,
        )

        # Run hyperparameter search
        best_trial = trainer.hyperparameter_search(
            direction="maximize",
            backend="optuna",
            hp_space=optuna_hp_space,
            hp_name=lambda trial: f"trial_{trial.number}",
            n_trials=args.n_trials,
            compute_objective=compute_objective,
        )

        print(f"\n\nBest trial parameters: {best_trial.hyperparameters}")
        print(f"Best trial F1 score: {best_trial.objective}")

        # Save best parameters to JSON file
        best_params_dict = best_trial.hyperparameters
        with open("best_params.json", "w") as f:
            json.dump(best_params_dict, f, indent=4)
        print("Best parameters saved to best_params.json")

        # Update training arguments with best parameters
        for param, value in best_trial.hyperparameters.items():
            if hasattr(training_args, param):
                setattr(training_args, param, value)

        # Initialize trainer with best parameters
        model = model_init()

    elif args.use_best_params:
        print("Using best parameters from previous hyperparameter optimization...")

        # Load best parameters from file
        try:
            with open("best_params.json", "r") as f:
                best_params_dict = json.load(f)
                for param, value in best_params_dict.items():
                    if hasattr(training_args, param):
                        setattr(training_args, param, value)
                        print(f"Setting {param} = {value}")
        except FileNotFoundError:
            print("Warning: best_params.json not found. Using default parameters.")

        # Initialize model
        model = model_init()

    else:
        # Standard training without HPO
        print("Training with default parameters...")
        model = RobertaForSequenceClassification.from_pretrained(
            model_name, num_labels=2, problem_type="single_label_classification"
        )

    # Initialize Trainer (or reinitialize with the model)
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
