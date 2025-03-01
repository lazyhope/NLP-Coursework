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
    n = len(words) // 2
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
    df_modified.loc[mask, "text"] = df_modified.loc[mask, "text"].apply(lambda x: synonym_replacement(x))
    return df_modified

def _back_translate(text):
    """
    Translate text to another language and back for augmentation.
    """
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**inputs)
    translated_text = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]

    # 目标语言 -> 英语
    model_rev = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
    tokenizer_rev = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
    inputs_rev = tokenizer_rev(translated_text, return_tensors="pt", padding=True, truncation=True)
    back_translated = model_rev.generate(**inputs_rev)
    back_translated_text = tokenizer_rev.batch_decode(back_translated, skip_special_tokens=True)[0]

    return back_translated_text

def apply_back_translation(df):
    """
    Apply back-translation only to new upsampled data.
    :param df: DataFrame with 'text' column
    :return: Augmented DataFrame
    """
    model_name = "Helsinki-NLP/opus-mt-en-fr"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    df_aug = df[df["new"] == 1].copy()  # Only new samples
    df_aug["text"] = df_aug["text"].apply(lambda x: _back_translate(x))
    df_aug["augmented"] = "back_translation"  # Augmentation type
    return df_aug