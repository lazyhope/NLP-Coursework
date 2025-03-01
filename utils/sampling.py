import pandas as pd
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import numpy as np

RATIO = 0.5

# Helper function to vectorize text data
def _vectorize_text(data):
    """
    Vectorize text data using TfidfVectorizer
    """
    vectorizer = TfidfVectorizer()
    X_tfidf = vectorizer.fit_transform(data["text"])
    y = data["label"]
    return X_tfidf, y, vectorizer

def _generate_text(X_resampled, vectorizer, X_original, text_data):
    """
    Generate text data from resampled data
    """
    nn = NearestNeighbors(n_neighbors=1, metric="cosine")
    nn.fit(X_original)

    indices = nn.kneighbors(X_resampled, return_distance=False)
    synthetic_texts = [text_data[i[0]] for i in indices]

    return synthetic_texts
    

# Sampling strategy

# 1) Copying
def copy_upsampling(data):
    pos = data[data["label"] == 1]
    neg = data[data["label"] == 0]
    ratio = int(len(neg) // len(pos) * RATIO)

    print(f"Copying minority samples with ratio: {ratio}...")
    pos = pd.concat([pos]*ratio, ignore_index=True)
    
    return pd.concat([pos, neg], ignore_index=True)

# 2) SMOTE
def smote_upsampling(data, RATIO=0.5):
    """
    Apply SMOTE upsampling and mark new samples as 'new=1'.
    :param data: pandas DataFrame with 'text' and 'label'
    :param RATIO: Desired sampling ratio
    :return: DataFrame with new column 'new' (1=new samples, 0=original data)
    """
    X_tfidf, y, vectorizer = _vectorize_text(data)

    smote = SMOTE(sampling_strategy=RATIO, random_state=42)
    print(f"Applying SMOTE with ratio: {RATIO}...")
    X_resampled, y_resampled = smote.fit_resample(X_tfidf, y)

    # Convert resampled data to text
    print(f"Converting resampled data to text...")
    new_texts = _generate_text(X_resampled[len(data):], vectorizer, X_tfidf, data["text"].values)

    # Create DataFrame for new samples
    new_data = pd.DataFrame({"text": new_texts, "label": y_resampled[len(data):]})
    new_data["new"] = 1  # New generated data

    # Original data
    original_data = data.copy()
    original_data["new"] = 0

    return pd.concat([original_data, new_data], ignore_index=True)

# 3) ADASYN
def adasyn_upsampling(data, RATIO=0.5):
    """
    Apply ADASYN upsampling and mark new samples as 'new=1'.
    :param data: pandas DataFrame with 'text' and 'label'
    :param RATIO: Desired sampling ratio
    :return: DataFrame with new column 'new' (1=new samples, 0=original data)
    """
    X_tfidf, y, vectorizer = _vectorize_text(data)

    adasyn = ADASYN(sampling_strategy=RATIO, random_state=42)
    print(f"Applying ADASYN with ratio: {RATIO}...")
    X_resampled, y_resampled = adasyn.fit_resample(X_tfidf, y)

    # Convert resampled data to text
    print(f"Converting resampled data to text...")
    new_texts = _generate_text(X_resampled[len(data):], vectorizer, X_tfidf, data["text"].values)

    # Create DataFrame for new samples
    new_data = pd.DataFrame({"text": new_texts, "label": y_resampled[len(data):]})
    new_data["new"] = 1  # New generated data

    # Original data
    original_data = data.copy()
    original_data["new"] = 0

    return pd.concat([original_data, new_data], ignore_index=True)