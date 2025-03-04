import random
import nltk
from nltk.corpus import wordnet
from transformers import MarianMTModel, MarianTokenizer

def _synonym_replacement(text):
    """
    Replace 'n' words in a sentence with their synonyms.
    :param text: Original sentence
    :param n: Number of words to replace
    :return: Augmented sentence
    """
    words = text.split()
    n = 10
    new_words = words.copy()
    
    # 获取有同义词的单词索引
    replaceable_indices = [i for i, word in enumerate(words) if wordnet.synsets(word)]
    
    # 如果没有可替换单词，直接返回原文本
    if not replaceable_indices:
        return text
    
    # 确保至少替换一个单词
    num_to_replace = min(n, len(replaceable_indices))
    selected_indices = random.sample(replaceable_indices, num_to_replace)

    for i in selected_indices:
        synonyms = wordnet.synsets(words[i])
        if synonyms:
            synonym = synonyms[0].lemmas()[0].name().replace("_", " ")
            new_words[i] = synonym

    return " ".join(new_words)

def apply_synonym_replacement(df):
    """
    Apply synonym replacement only to new upsampled data.
    :param df: DataFrame with 'text' column
    :return: Augmented DataFrame
    """
    # download wordnet
    nltk.download("wordnet")

    df_modified = df.copy()  # Copy DataFrame
    mask = df_modified["new"] == 1  # Only new samples
    df_modified.loc[mask, "text"] = df_modified.loc[mask, "text"].apply(lambda x: _synonym_replacement(x))
    return df_modified

def apply_back_translation(df, src_lang="en", tgt_lang="fr", text_col="text"):
    """
    Apply back-translation only to rows with new==1 in the DataFrame.
    Loads translation models for src_lang->tgt_lang and tgt_lang->src_lang,
    applies back-translation on the specified text column for rows where "new" equals 1,
    and sets par_id for these augmented rows to 0. Optionally, save the augmented DataFrame as CSV.
    
    Args:
        df (pd.DataFrame): DataFrame containing at least the 'text' and 'new' columns.
        src_lang (str): Source language code (default "en").
        tgt_lang (str): Target language code (default "fr").
        text_col (str): Name of the text column (default "text").
        par_id_col (str): Name of the ID column (default "par_id").
        output_csv (str): File path to save the augmented DataFrame (default None).
    
    Returns:
        pd.DataFrame: Augmented DataFrame with back-translated text for rows with new==1.
    """
    df_aug = df.copy()
    
    # Load the translation model for src_lang -> tgt_lang
    print(f"Loading translation model for {src_lang} -> {tgt_lang}...")
    model_name_fwd = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    tokenizer_fwd = MarianTokenizer.from_pretrained(model_name_fwd)
    model_fwd = MarianMTModel.from_pretrained(model_name_fwd)
    
    # Load the translation model for tgt_lang -> src_lang
    print(f"Loading translation model for {tgt_lang} -> {src_lang}...")
    model_name_bwd = f"Helsinki-NLP/opus-mt-{tgt_lang}-{src_lang}"
    tokenizer_bwd = MarianTokenizer.from_pretrained(model_name_bwd)
    model_bwd = MarianMTModel.from_pretrained(model_name_bwd)
    
    def _back_translate(text: str) -> str:
        """
        Translate text from src_lang to tgt_lang and back to src_lang.
        """
        if not isinstance(text, str) or text.strip() == "":
            return text
        # Translate from src_lang to tgt_lang
        inputs_fwd = tokenizer_fwd(text, return_tensors="pt", padding=True, truncation=True)
        translated = model_fwd.generate(**inputs_fwd)
        translated_text = tokenizer_fwd.batch_decode(translated, skip_special_tokens=True)[0]
        
        # Translate back from tgt_lang to src_lang
        inputs_bwd = tokenizer_bwd(translated_text, return_tensors="pt", padding=True, truncation=True)
        back_translated = model_bwd.generate(**inputs_bwd)
        back_translated_text = tokenizer_bwd.batch_decode(back_translated, skip_special_tokens=True)[0]
        return back_translated_text

    # Apply back-translation only to rows with new==1
    mask = df_aug["new"] == 1
    df_aug.loc[mask, text_col] = df_aug.loc[mask, text_col].apply(_back_translate)
    
    return df_aug

def add_keyword_prefix(df, keyword_col="keyword", text_col="text", new_text_col="combined_text"):
    """
    Concatenate the keyword as a prefix to the text.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing at least the columns [keyword, text].
        keyword_col (str): The column name for keywords.
        text_col (str): The column name for the text.
        new_text_col (str): The column name for the combined text.
        
    Returns:
        pd.DataFrame: A new DataFrame with an additional column 'combined_text'
                      in which each row is formatted as "[keyword] text".
    """
    df_copy = df.copy()
    df_copy[new_text_col] = "<" + df_copy[keyword_col].astype(str) + "> " + df_copy[text_col].astype(str)
    return df_copy
