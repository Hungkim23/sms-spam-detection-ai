import os
import pandas as pd

def load_data(file_path="data/spam.csv"):
    """
    Loads and cleans the initial dataset structure.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError("Dataset not found. Please place spam.csv inside the data/ folder.")
    
    # Kaggle dataset often needs latin-1 encoding
    df = pd.read_csv(file_path, encoding='latin-1')
    
    # Rename columns flexibly
    if 'v1' in df.columns and 'v2' in df.columns:
        df = df.rename(columns={'v1': 'label', 'v2': 'message'})
    
    # Drop Unnamed columns
    unnamed_cols = [col for col in df.columns if 'Unnamed' in col]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
        
    # Keep only label and message
    if 'label' in df.columns and 'message' in df.columns:
        df = df[['label', 'message']]
    
    return df

def prepare_dataset(df):
    """
    Cleans data, drops NAs, duplicates, and adds metadata columns.
    """
    df = df.dropna(subset=['label', 'message']).copy()
    df = df.drop_duplicates(subset=['message']).copy()
    
    df['label'] = df['label'].str.lower()
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
    
    df['message_length'] = df['message'].apply(lambda x: len(str(x)))
    df['word_count'] = df['message'].apply(lambda x: len(str(x).split()))
    
    return df
