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
# 1. Load Data
# ------------------------------
def load_data():
    """
    Load train, validation, and test data from CSV files (with headers).
    """
    TRAIN_PATH = "data/train_data.csv"
    VAL_PATH   = "data/val_data.csv"
    TEST_PATH  = "data/test_data.csv"
    train_df = pd.read_csv(TRAIN_PATH)
    val_df   = pd.read_csv(VAL_PATH)
    test_df  = pd.read_csv(TEST_PATH)
    X_train, y_train = train_df["text"].tolist(), train_df["label"].tolist()
    X_val,   y_val   = val_df["text"].tolist(),   val_df["label"].tolist()
    X_test,  y_test  = test_df["text"].tolist(),  test_df["label"].tolist()
    print(f"Training examples: {len(X_train)}")
    print(f"Validation examples: {len(X_val)}")
    print(f"Test examples: {len(X_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test

# ------------------------------
# 2. Compute Metrics Function
# ------------------------------
def compute_metrics(true_labels, predictions):
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average="binary", zero_division=0
    )
    acc = accuracy_score(true_labels, predictions)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# ------------------------------
# 3. Word2VecVectorizer
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
# 4. Load Word2Vec Model
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
# 5. Training Loop with Global Step Evaluation Every 100 Steps
# ------------------------------
def train_word2vec_logistic():
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    
    # Load Word2Vec model and create vectorizer.
    w2v_model = load_word2vec_model()
    vectorizer = Word2VecVectorizer(w2v_model, embedding_dim=300)
    
    print("Transforming training data...")
    X_train_emb = vectorizer.transform(X_train)
    print("Transforming validation data...")
    X_val_emb = vectorizer.transform(X_val)
    print("Transforming test data...")
    X_test_emb = vectorizer.transform(X_test)
    
    # Shuffle training data (once, then shuffle each epoch separately)
    X_train_emb, y_train = shuffle(X_train_emb, y_train, random_state=42)
    
    # Initialize SGDClassifier for logistic regression (using loss="log_loss")
    clf = SGDClassifier(loss="log_loss", max_iter=1, tol=None, random_state=42)
    classes = np.unique(y_train)
    
    n_epochs = 10
    batch_size = 64
    global_step = 0
    global_steps = []
    train_f1_list = []
    val_f1_list = []
    
    for epoch in range(n_epochs):
        print(f"Epoch {epoch+1}/{n_epochs}")
        # Shuffle training data at the beginning of each epoch
        X_train_shuffled, y_train_shuffled = shuffle(X_train_emb, y_train, random_state=epoch)
        
        # Process mini-batches for this epoch
        n_batches = int(np.ceil(X_train_shuffled.shape[0] / batch_size))
        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size
            X_batch = X_train_shuffled[start:end]
            y_batch = y_train_shuffled[start:end]
            if global_step == 0:
                clf.partial_fit(X_batch, y_batch, classes=classes)
            else:
                clf.partial_fit(X_batch, y_batch)
            global_step += 1
            
            # Evaluate every 100 global steps
            if global_step % 100 == 0:
                y_train_pred = clf.predict(X_train_emb)
                y_val_pred = clf.predict(X_val_emb)
                train_f1 = f1_score(y_train, y_train_pred, average="binary", zero_division=0)
                val_f1 = f1_score(y_val, y_val_pred, average="binary", zero_division=0)
                train_f1_list.append(train_f1)
                val_f1_list.append(val_f1)
                global_steps.append(global_step)
                print(f"Global Step {global_step}: Training F1 = {train_f1:.3f}, Validation F1 = {val_f1:.3f}")
    
    # Plot learning curve (F1 vs. Global Steps)
    plt.figure(figsize=(8,6))
    plt.plot(global_steps, train_f1_list, marker="o", label="Training F1")
    plt.plot(global_steps, val_f1_list, marker="o", label="Validation F1")
    plt.xlabel("Global Step", fontsize=16)
    plt.ylabel("F1 Score", fontsize=16)
    plt.title("Learning Curve (F1 Score) - Word2Vec + Logistic Regression", fontsize=18)
    plt.ylim(0,1)
    plt.grid(True)
    plt.legend(fontsize=16)
    plt.savefig("figures/Word2Vec_learning_curve.png")
    plt.show()
    
    # Final Evaluation on Validation Set
    print("\n=== Final Validation Performance ===")
    y_val_pred = clf.predict(X_val_emb)
    print(classification_report(y_val, y_val_pred, digits=3))
    
    # Final Evaluation on Test Set
    print("\n=== Test Performance ===")
    y_test_pred = clf.predict(X_test_emb)
    print(classification_report(y_test, y_test_pred, digits=3))
    
    val_metrics = compute_metrics(y_val, y_val_pred)
    test_metrics = compute_metrics(y_test, y_test_pred)
    print("Validation metrics:", val_metrics)
    print("Test metrics:", test_metrics)

if __name__ == "__main__":
    train_word2vec_logistic()
