import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

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
        true_labels, predictions, average="binary"
    )
    acc = accuracy_score(true_labels, predictions)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def main():
    # Load the data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    # Create a pipeline that transforms text into a BoW representation,
    # then applies MultinomialNB classifier
    model_pipeline = Pipeline([
        ("vect", CountVectorizer()),        # Converts text to a matrix of token counts
        ("clf", MultinomialNB())            # Naive Bayes classifier
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


    metrics = compute_metrics(y_test, y_test_pred)
    print("Overall metrics:", metrics)

if __name__ == "__main__":
    main()
