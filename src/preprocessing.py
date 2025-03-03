import pandas as pd
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def load_data(file_path, sep=';'):
    return pd.read_csv(file_path, sep=sep)

def preprocess_data(df, speaker='Harry'):
    df = df[df['Character'] == speaker]
    df['Sentence'] = df['Sentence'].str.lower().str.strip()
    df['Sentence'] = df['Sentence'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))
    # custom_stop_words = ENGLISH_STOP_WORDS - {'harry', 'potter'}
    # df['Sentence'] = df['Sentence'].apply(
    #     lambda x: ' '.join([word for word in x.split() if word not in custom_stop_words])
    # )
    return df.dropna().drop_duplicates()