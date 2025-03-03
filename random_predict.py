import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

def load_test_data():
    TEST_PATH = "data/test_data.csv"
    test_df = pd.read_csv(TEST_PATH)
    y_test = test_df["label"].tolist()
    return y_test

# Load true test labels
y_test = load_test_data()

# Set a random seed for reproducibility
np.random.seed(42)

# Randomly predict class labels (0 or 1) for each test example
random_predictions = np.random.choice([0, 1], size=len(y_test))

# Calculate the F1 score for the positive class (class 1)
f1 = f1_score(y_test, random_predictions, average="binary", pos_label=1)
print("Random prediction F1 score for class 1:", f1)
