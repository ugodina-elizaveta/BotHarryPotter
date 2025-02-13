import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df, speaker='JOEY'):
    # Оставляем только реплики нужного персонажа
    df = df[df['Speaker'] == speaker]
    df['Text'] = df['Text'].str.lower().str.strip()
    return df