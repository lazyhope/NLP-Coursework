import re
import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
import pandas as pd

# Ensure the required NLTK packages are downloaded
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('punkt_tab')

def _lemmatize_text(text: str) -> str:
    """
    Lemmatize the input text using WordNetLemmatizer.
    """
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return " ".join(lemmatized_tokens)

def _stem_text(text: str) -> str:
    """
    Stem the input text using PorterStemmer.
    """
    stemmer = PorterStemmer()
    tokens = word_tokenize(text)
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return " ".join(stemmed_tokens)

def preprocess_dataframe(df: pd.DataFrame, text_col: str = "text", method: str = "lemmatize") -> pd.DataFrame:
    """
    Apply text preprocessing (lemmatization or stemming) directly on the specified text column of the entire DataFrame.
    The original text column is modified (overwritten) with the processed text.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        text_col (str): Column name containing text.
        method (str): "lemmatize" or "stem" (default is "lemmatize").
        
    Returns:
        pd.DataFrame: A new DataFrame with the text column processed.
    """
    df_clean = df.copy()
    # Ensure the text column is treated as string
    df_clean[text_col] = df_clean[text_col].astype(str)
    
    if method == "lemmatize":
        df_clean[text_col] = df_clean[text_col].apply(_lemmatize_text)
    elif method == "stem":
        df_clean[text_col] = df_clean[text_col].apply(_stem_text)
    else:
        raise ValueError("Unknown method: choose 'lemmatize' or 'stem'")
    
    return df_clean

if __name__ == "__main__":
    # Load the entire DataFrame from CSV
    df = pd.read_csv("data/train_data.csv")
    print(f"Loaded data with shape: {df.shape}")
    
    # Choose preprocessing method: "lemmatize" or "stem"
    method = "lemmatize"  # 修改为 "stem" 则进行词干提取
    df_preprocessed = preprocess_dataframe(df, text_col="text", method=method)
    
    # Display the first few rows for comparison
    print("Original text samples:")
    print(df["text"].head(5))
    print("\nProcessed text samples:")
    print(df_preprocessed["text"].head(5))
    
    # Save the processed DataFrame to a new CSV file
    df_preprocessed.to_csv("data/train_data_preprocessed.csv", index=False)
    print("Preprocessed data saved to data/train_data_preprocessed.csv")