import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score, classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.utils import shuffle

def load_data():
    """Load train, validation, and test data from CSV files."""
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

def compute_metrics(true_labels, predictions):
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average="binary", zero_division=0
    )
    acc = accuracy_score(true_labels, predictions)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

def train_bow_logistic_steps():
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    
    # Create CountVectorizer and transform text data into BoW representation
    vectorizer = CountVectorizer()
    X_train_mat = vectorizer.fit_transform(X_train)
    X_val_mat = vectorizer.transform(X_val)
    X_test_mat = vectorizer.transform(X_test)
    
    # Shuffle training data
    X_train_mat, y_train = shuffle(X_train_mat, y_train, random_state=42)
    
    # Initialize SGDClassifier with logistic regression loss ("log_loss")
    clf = SGDClassifier(loss="log_loss", max_iter=1, tol=None, random_state=42)
    classes = np.unique(y_train)
    
    batch_size = 64
    global_step = 0
    global_steps = []
    train_f1_list = []
    val_f1_list = []
    
    n_epochs = 10
    # Training loop: iterate over epochs, but record metrics every 100 steps.
    for epoch in range(n_epochs):
        print(f"Epoch {epoch+1}/{n_epochs}")
        # Shuffle the training data at the beginning of each epoch
        X_train_shuffled, y_train_shuffled = shuffle(X_train_mat, y_train, random_state=epoch)
        # Process mini-batches in this epoch
        for i in range(0, X_train_shuffled.shape[0], batch_size):
            X_batch = X_train_shuffled[i:i+batch_size]
            y_batch = y_train_shuffled[i:i+batch_size]
            if global_step == 0:
                clf.partial_fit(X_batch, y_batch, classes=classes)
            else:
                clf.partial_fit(X_batch, y_batch)
            global_step += 1
            # Every 100 global steps, evaluate on the full training and validation sets
            if global_step % 100 == 0:
                y_train_pred = clf.predict(X_train_mat)
                y_val_pred = clf.predict(X_val_mat)
                train_f1 = f1_score(y_train, y_train_pred, average="binary", zero_division=0)
                val_f1 = f1_score(y_val, y_val_pred, average="binary", zero_division=0)
                train_f1_list.append(train_f1)
                val_f1_list.append(val_f1)
                global_steps.append(global_step)
                print(f"Global Step {global_step}: Training F1 = {train_f1:.3f}, Validation F1 = {val_f1:.3f}")
    
    # Plot the learning curve (F1 vs. Global Steps)
    plt.figure(figsize=(8,6))
    plt.plot(global_steps, train_f1_list, marker="o", label="Training F1")
    plt.plot(global_steps, val_f1_list, marker="o", label="Validation F1")
    plt.xlabel("Global Step", fontsize=16)
    plt.ylabel("F1 Score", fontsize=16)
    plt.title("Learning Curve (F1 Score) - BoW + Logistic Regression", fontsize=18)
    plt.ylim(0,1)
    plt.grid(True)
    plt.legend(fontsize=16)
    plt.savefig("figures/BoW_logistic_learning_curve.png")
    plt.show()
    
    # --- Final Evaluation ---
    print("\n=== Final Validation Performance ===")
    y_val_pred = clf.predict(X_val_mat)
    print(classification_report(y_val, y_val_pred, digits=3))
    
    print("\n=== Test Performance ===")
    y_test_pred = clf.predict(X_test_mat)
    print(classification_report(y_test, y_test_pred, digits=3))
    
    val_metrics = compute_metrics(y_val, y_val_pred)
    test_metrics = compute_metrics(y_test, y_test_pred)
    print("Validation metrics:", val_metrics)
    print("Test metrics:", test_metrics)

if __name__ == "__main__":
    train_bow_logistic_steps()
