import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gensim.downloader as api  # Built-in downloader for pre-trained models

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (classification_report, accuracy_score,
                             precision_recall_fscore_support, f1_score)
from sklearn.utils import shuffle

# ------------------------------
# 1. File Paths
# ------------------------------
TRAIN_PATH = "data/train_data.csv"
VAL_PATH   = "data/val_data.csv"
TEST_PATH  = "data/test_data.csv"

# ------------------------------
# 2. Load Data
# ------------------------------
def load_data():
    """
    Load train, validation, and test data from CSV files (with headers).
    """
    print("Loading data...")
    train_df = pd.read_csv(TRAIN_PATH)
    val_df   = pd.read_csv(VAL_PATH)
    test_df  = pd.read_csv(TEST_PATH)

    # Convert labels to int if needed
    train_df["label"] = train_df["label"].astype(int)
    val_df["label"]   = val_df["label"].astype(int)
    test_df["label"]  = test_df["label"].astype(int)

    X_train, y_train = train_df["text"].tolist(), train_df["label"].tolist()
    X_val,   y_val   = val_df["text"].tolist(),   val_df["label"].tolist()
    X_test,  y_test  = test_df["text"].tolist(),  test_df["label"].tolist()

    print(f"Training examples: {len(X_train)}")
    print(f"Validation examples: {len(X_val)}")
    print(f"Test examples: {len(X_test)}")

    return X_train, y_train, X_val, y_val, X_test, y_test

# ------------------------------
# 3. Load Word2Vec using Gensim's Built-In Downloader
# ------------------------------
def load_word2vec_model():
    """
    Download and load a pre-trained Word2Vec model using gensim's built-in downloader.
    This downloads the Google News vectors ("word2vec-google-news-300").
    """
    print("Downloading Word2Vec model using gensim's built-in downloader...")
    w2v_model = api.load("word2vec-google-news-300")
    print(f"Vocabulary size: {len(w2v_model.key_to_index)} words.")
    return w2v_model

# ------------------------------
# 4. Word2VecVectorizer
# ------------------------------
class Word2VecVectorizer(BaseEstimator, TransformerMixin):
    """
    Transforms a list of texts into average Word2Vec embeddings.
    """
    def __init__(self, w2v_model, embedding_dim=300):
        self.w2v_model = w2v_model
        self.embedding_dim = embedding_dim

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_emb = []
        for text in X:
            words = text.split()  # simple whitespace split
            word_vectors = []
            for w in words:
                w_lower = w.lower()
                if w_lower in self.w2v_model:
                    word_vectors.append(self.w2v_model[w_lower])
            if word_vectors:
                avg_vector = np.mean(word_vectors, axis=0)
            else:
                avg_vector = np.zeros(self.embedding_dim)
            X_emb.append(avg_vector)
        return np.array(X_emb)

# ------------------------------
# 5. Compute Metrics
# ------------------------------
def compute_metrics(true_labels, predictions):
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average="binary", zero_division=0
    )
    acc = accuracy_score(true_labels, predictions)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# ------------------------------
# 6. Main Training and Learning Curve Plotting
# ------------------------------
def main():
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    # Shuffle training data
    X_train, y_train = shuffle(X_train, y_train, random_state=42)

    # Load Word2Vec model
    w2v_model = load_word2vec_model()

    # Instantiate the vectorizer and transform the data once
    vectorizer = Word2VecVectorizer(w2v_model, embedding_dim=300)
    print("Transforming training data...")
    X_train_emb = vectorizer.transform(X_train)
    print("Transforming validation data...")
    X_val_emb = vectorizer.transform(X_val)
    print("Transforming test data...")
    X_test_emb = vectorizer.transform(X_test)

    # Initialize SGDClassifier for logistic regression using partial_fit
    # (SGDClassifier with loss="log_loss" behaves like logistic regression)
    clf = SGDClassifier(loss="log_loss", max_iter=1, tol=None, random_state=42)
    classes = np.array([0, 1])
    n_epochs = 10
    train_f1_per_epoch = []
    val_f1_per_epoch = []

    # Training loop: iterate over epochs using partial_fit
    for epoch in range(n_epochs):
        print(f"Epoch {epoch+1}/{n_epochs}")
        clf.partial_fit(X_train_emb, y_train, classes=classes)
        
        # Evaluate on training set
        y_train_pred = clf.predict(X_train_emb)
        train_f1 = f1_score(y_train, y_train_pred, average="binary", zero_division=0)
        train_f1_per_epoch.append(train_f1)
        
        # Evaluate on validation set
        y_val_pred = clf.predict(X_val_emb)
        val_f1 = f1_score(y_val, y_val_pred, average="binary", zero_division=0)
        val_f1_per_epoch.append(val_f1)
        
        print(f"Training F1: {train_f1:.3f} | Validation F1: {val_f1:.3f}")

    # Plot the F1 learning curve (Training and Validation F1 vs. Epoch)
    epochs = np.arange(1, n_epochs+1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_f1_per_epoch, marker="o", label="Training F1 Score")
    plt.plot(epochs, val_f1_per_epoch, marker="o", label="Validation F1 Score")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title('Learning Curve (F1 Score) of Word2Vec + Logistic Regression')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.savefig("Word2Vec_learning_curve.png")
    plt.show()

    # Print final validation performance report
    print("\n=== Final Validation Performance ===")
    y_val_pred = clf.predict(X_val_emb)
    print(classification_report(y_val, y_val_pred, digits=3))

    # Evaluate final model on test set
    print("\n=== Test Performance ===")
    y_test_pred = clf.predict(X_test_emb)
    print(classification_report(y_test, y_test_pred, digits=3))
    test_metrics = compute_metrics(y_test, y_test_pred)
    print("Overall test metrics:", test_metrics)

if __name__ == "__main__":
    main()
