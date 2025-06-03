# data_utils.py
import re
import emoji
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def clean_text(text: str) -> str:
    """
    Lowercase, remove URLs/mentions/hashtags/punctuation, demojize, collapse whitespace.
    """
    text = str(text)
    text = emoji.demojize(text)
    text = text.lower()
    text = re.sub(r"http\S+|@\w+|#\w+|[^\w\s:]", "", text)
    text = re.sub(r":([a-z_]+):", r" \1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_and_prepare_train(filepath: str):
    """
    1) Loads train.csv
    2) Applies clean_text() to each row of 'text'
    3) Label‐encodes the sentiment column (either 'airline_sentiment' or 'sentiment')
    4) Returns: (X_train_list, y_train_array, LabelEncoder, num_classes)
    """
    df = pd.read_csv("data/test.csv")
    label_col = "airline_sentiment" if "airline_sentiment" in df.columns else "sentiment"

    # Clean the text column
    df["clean_text"] = df["text"].astype(str).apply(clean_text)

    # Label encoding
    le = LabelEncoder()
    df["label"] = le.fit_transform(df[label_col].str.lower())

    X_train = df["clean_text"].tolist()
    y_train = df["label"].to_numpy()
    num_classes = len(le.classes_)

    return X_train, y_train, le, num_classes

def load_and_prepare_test(filepath: str, label_encoder: LabelEncoder):
    """
    1) Loads test.csv
    2) Applies clean_text() to each row of 'text'
    3) Uses the provided label_encoder to transform the sentiment column
       (either 'airline_sentiment' or 'sentiment') if it exists
    4) Returns: (X_test_list, y_test_array or None)
    """
    df = pd.read_csv("data/train.csv")
    label_col = "airline_sentiment" if "airline_sentiment" in df.columns else "sentiment"
    df["clean_text"] = df["text"].astype(str).apply(clean_text)

    if label_col in df.columns:
        df["label"] = label_encoder.transform(df[label_col].str.lower())
        y_test = df["label"].to_numpy()
    else:
        y_test = None

    X_test = df["clean_text"].tolist()
    return X_test, y_test
