import pandas as pd
import re
from datasets import load_dataset, Dataset, concatenate_datasets

jfs_df = pd.read_csv("dataset/jakarta_field_station_full_sentence.csv")
tapaleuk_df = pd.read_csv("dataset/tapaleuk_full_sentence.csv")
puisi_df = pd.read_csv("dataset/puisi.csv", encoding='latin-1')
tapaleuk_new_df = pd.read_csv("dataset/tapaleuk_new.csv")
fineweb_df = load_dataset("HuggingFaceFW/fineweb-2", name="mkn_Latn", split="train+test").to_pandas()


def preprocessing(text):
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r"(?<!\w)xx\b", "", text)
    text = re.sub(r'0\.{1,}', '', text)
    text = re.sub(r'\w+\([^\)]*\)', '', text)
    text = re.sub(r'\bx{1,}\b', '', text)
    text = re.sub(r'\(([^)]+)\)', r'\1', text)
    text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text)
    text = re.sub(r'\.\s\.', '', text)
    text = re.sub(r'\b0\b', '', text)

    # untuk fineweb
    text = re.sub(r'[|\d*]+', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r':\s*-\s*:?', '', text)
    text = re.sub(r'[:;]+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def load_data():
    jfs_df['mkn_sentence_full'] = jfs_df['mkn_sentence_full'].apply(lambda x: preprocessing(x))
    tapaleuk_df['mkn_sentence_full'] = tapaleuk_df['mkn_sentence_full'].apply(lambda x: preprocessing(x))
    fineweb_df['text'] = fineweb_df['text'].apply(lambda x: preprocessing(x))

    jfs_dataset = Dataset.from_pandas(jfs_df)
    jfs_dataset = jfs_dataset.rename_columns({"mkn_sentence_full": "text"})
    jfs_dataset = jfs_dataset.remove_columns(["eng_sentence_full"])
    tapaleuk_dataset = Dataset.from_pandas(tapaleuk_df)
    tapaleuk_dataset = tapaleuk_dataset.rename_columns({"mkn_sentence_full": "text"})
    tapaleuk_dataset = tapaleuk_dataset.remove_columns(["eng_sentence_full"])
    puisi_dataset = Dataset.from_pandas(puisi_df)
    puisi_dataset = puisi_dataset.rename_columns({"text": "text"})
    tapaleuk_new_dataset = Dataset.from_pandas(tapaleuk_new_df)
    tapaleuk_new_dataset = tapaleuk_new_dataset.rename_columns({"text": "text"})
    fineweb_dataset = Dataset.from_pandas(fineweb_df)

    document = concatenate_datasets([fineweb_dataset, jfs_dataset, tapaleuk_dataset, puisi_dataset, tapaleuk_new_dataset])

    return document
