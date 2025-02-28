import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (classification_report, accuracy_score, 
                             precision_recall_fscore_support)
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.utils import shuffle

# Paths to your CSV files
TRAIN_PATH = "data/train_data.csv"
VAL_PATH   = "data/val_data.csv"
TEST_PATH  = "data/test_data.csv"

def load_data():
    """Load train, validation, and test data from CSV files."""
    print("Loading data...")
    train_df = pd.read_csv(TRAIN_PATH)
    val_df   = pd.read_csv(VAL_PATH)
    test_df  = pd.read_csv(TEST_PATH)

    # Extract features (X) and labels (y)
    X_train, y_train = train_df["text"], train_df["label"]
    X_val,   y_val   = val_df["text"],   val_df["label"]
    X_test,  y_test  = test_df["text"],  test_df["label"]

    print(f"Training examples: {len(X_train)}")
    print(f"Validation examples: {len(X_val)}")
    print(f"Test examples: {len(X_test)}")

    return X_train, y_train, X_val, y_val, X_test, y_test

def compute_metrics(true_labels, predictions):
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average="binary", zero_division=0
    )
    acc = accuracy_score(true_labels, predictions)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, X, y, filename="BoW_learning_curve.png"):
    # Using 5-fold cross-validation and training sizes from 10% to 100% of data,
    # and scoring based on F1 score (binary)
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=5, scoring='f1',
        train_sizes=np.linspace(0.1, 1.0, 10), verbose=1, n_jobs=-1
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_scores_mean, label='Training F1 Score', marker='o')
    plt.plot(train_sizes, val_scores_mean, label='Validation F1 Score', marker='o')
    plt.xlabel("Epoch")
    plt.ylabel('F1 Score')
    plt.title('Learning Curve (F1 Score) of BoW + Naive Bayes')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.show()


def main():
    # Load the data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    # Shuffle the training data
    X_train, y_train = shuffle(X_train, y_train, random_state=42)

    # Create a pipeline that transforms text into a BoW representation,
    # then applies a MultinomialNB classifier
    model_pipeline = Pipeline([
        ("vect", CountVectorizer()),  # Converts text to a matrix of token counts
        ("clf", MultinomialNB())        # Naive Bayes classifier
    ])

    # Train on the training data
    model_pipeline.fit(X_train, y_train)

    # Evaluate on the validation data
    print("\n=== Validation Performance ===")
    y_val_pred = model_pipeline.predict(X_val)
    print(classification_report(y_val, y_val_pred, digits=3))

    # Evaluate on the test data
    print("\n=== Test Performance ===")
    y_test_pred = model_pipeline.predict(X_test)
    print(classification_report(y_test, y_test_pred, digits=3))

    # Compute overall metrics
    val_metrics = compute_metrics(y_val, y_val_pred)
    test_metrics = compute_metrics(y_test, y_test_pred)
    print("Validation metrics:", val_metrics)
    print("Test metrics:", test_metrics)

    # Plot F1 scores
    plot_learning_curve(model_pipeline, X_train, y_train)

if __name__ == "__main__":
    main()
