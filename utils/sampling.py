import pandas as pd
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import numpy as np


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

# Copying
def copy_upsampling(data, RATIO=0.5):
    """
    Upsample the minority class (label 1) so that its total count becomes half of the majority class (label 0).
    The newly sampled instances are marked with new=1, while original instances are marked with new=0.
    
    Args:
        data (pd.DataFrame): Input DataFrame containing at least the "label" column.
        RATIO (float): The target ratio of minority samples to majority samples (default 0.5).
    
    Returns:
        pd.DataFrame: DataFrame with upsampled minority class and a new column "new".
    """
    # Separate the data into minority (label 1) and majority (label 0) samples
    pos = data[data["label"] == 1].copy()
    neg = data[data["label"] == 0].copy()
    
    # Mark all original samples with new = 0
    data_copy = data.copy()
    data_copy["new"] = 0
    
    # Compute target number for minority class: RATIO * number of majority samples
    target_count = int(len(neg) * RATIO)
    
    # If the current minority count is less than the target, sample additional rows from minority class
    if len(pos) < target_count:
        additional_needed = target_count - len(pos)
        pos_additional = pos.sample(n=additional_needed, replace=True, random_state=42)
        pos_additional["new"] = 1  # Mark newly sampled data as new
        # Combine the original pos with the additional samples
        pos_upsampled = pd.concat([pos, pos_additional], ignore_index=True)
    else:
        pos_upsampled = pos.copy()
    
    # Combine upsampled minority samples with majority samples (both already have new=0 for original majority)
    final_df = pd.concat([pos_upsampled, neg], ignore_index=True)
    return final_df

# Down Sampling
def downsampling(data):
    pos = data[data["label"] == 1]
    neg = data[data["label"] == 0]
    ratio = len(pos) * 2

    print(f"Selecting the first {ratio} majority samples...")
    neg = neg.iloc[:ratio]
    
    return pd.concat([pos, neg], ignore_index=True)

# SMOTE
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

# ADASYN
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