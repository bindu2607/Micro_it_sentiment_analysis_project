import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

import transformers
transformers.logging.set_verbosity_error()

import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import TFBertForSequenceClassification, BertTokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------- CONFIG -------------------
TEST_PATH = "data/test.csv"
SEQ_LEN = 50
BERT_MAX_LEN = 128
BERT_BATCH_SIZE = 8

def clean_text(text):
    import re, emoji
    text = str(text)
    text = emoji.demojize(text)
    text = text.lower()
    text = re.sub(r"http\S+|@\w+|#\w+|[^\w\s:]", "", text)
    text = re.sub(r":([a-z_]+):", r" \1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def ensure_dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_label_encoder():
    df = pd.read_csv("data/train.csv")
    label_col = "airline_sentiment" if "airline_sentiment" in df.columns else "sentiment"
    le = LabelEncoder()
    le.fit(df[label_col].str.lower())
    return le

def load_and_prepare_test(path):
    df = pd.read_csv(path)
    df["clean_text"] = df["text"].astype(str).apply(clean_text)
    texts = df["clean_text"].tolist()
    return texts, df

def plot_confusion_matrix(y_true, y_pred, label_names, model_name):
    cm = confusion_matrix(y_true, y_pred)
    folder = f"model/{model_name}"
    ensure_dir_exists(folder)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names)
    plt.title(f"{model_name.upper()} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    save_path = os.path.join(folder, "confusion_matrix.png")
    plt.savefig(save_path)
    plt.close()

def plot_metrics_bars(y_true, y_pred, label_names, model_name):
    folder = f"model/{model_name}"
    ensure_dir_exists(folder)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(label_names)), zero_division=0
    )
    acc_per_class = []
    cm = confusion_matrix(y_true, y_pred, labels=range(len(label_names)))
    for i in range(len(label_names)):
        acc = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
        acc_per_class.append(acc)
    metrics = {
        "Accuracy": acc_per_class,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    }
    colors = {
        "Accuracy": "#4C72B0",
        "Precision": "#DD8452",
        "Recall": "#55A868",
        "F1-score": "#C44E52"
    }
    for metric, values in metrics.items():
        plt.figure(figsize=(8, 5))
        bars = plt.bar(label_names, values, color=colors[metric])
        plt.ylim(0, 1)
        plt.ylabel(metric, fontsize=12)
        plt.title(f"{model_name.upper()} {metric} per Class", fontsize=14)
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., 
                     height + 0.02,
                     f'{height:.2f}',
                     ha='center', va='bottom')
        plt.tight_layout()
        save_path = os.path.join(folder, f"{model_name.replace('/','_')}_{metric.lower()}_bar.png")
        plt.savefig(save_path, dpi=120)
        plt.close()

def evaluate(y_true, y_pred, label_names, model_name):
    folder = f"model/{model_name}"
    ensure_dir_exists(folder)
    print(f"\n=== {model_name.upper()} ===")
    report = classification_report(y_true, y_pred, 
                                  target_names=label_names, 
                                  digits=4, 
                                  zero_division=0)
    print("\n📊 Classification Report:\n")
    print(report)
    acc = accuracy_score(y_true, y_pred)
    print(f"✅ Overall Accuracy: {acc:.4f}")
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    print(f"🎯 Weighted Precision: {precision:.4f}")
    print(f"🎯 Weighted Recall:    {recall:.4f}")
    print(f"🎯 Weighted F1-score:  {f1:.4f}")
    metrics_path = os.path.join(folder, "metrics_report.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Classification Report:\n{report}\n")
        f.write(f"\nOverall Accuracy: {acc:.4f}")
        f.write(f"\nWeighted Precision: {precision:.4f}")
        f.write(f"\nWeighted Recall: {recall:.4f}")
        f.write(f"\nWeighted F1-score: {f1:.4f}")
    plot_confusion_matrix(y_true, y_pred, label_names, model_name)
    plot_metrics_bars(y_true, y_pred, label_names, model_name)

def bert_predict_in_batches(model, tokenizer, texts, batch_size=8, max_len=128):
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        enc = tokenizer(batch_texts, 
                       truncation=True, 
                       padding="max_length", 
                       max_length=max_len, 
                       return_tensors="tf")
        logits = model(enc).logits
        probs = tf.nn.softmax(logits, axis=1).numpy()
        all_probs.append(probs)
    return np.vstack(all_probs)

def test_bert(le, label_names, texts, df):
    print("\n🧪 Testing BERT model...")
    tokenizer = BertTokenizer.from_pretrained("model/bert/bert_tokenizer")
    model = TFBertForSequenceClassification.from_pretrained("model/bert/bert_model")
    probs = bert_predict_in_batches(model, tokenizer, texts, 
                                   batch_size=BERT_BATCH_SIZE, 
                                   max_len=BERT_MAX_LEN)
    label_col = "airline_sentiment" if "airline_sentiment" in df.columns else "sentiment"
    y_true = le.transform(df[label_col].str.lower())
    y_pred = np.argmax(probs, axis=1)
    evaluate(y_true, y_pred, label_names, "bert")

def test_lstm(le, label_names, texts, df):
    print("\n🧪 Testing BiLSTM model...")
    with open("model/lstm/tokenizer_lstm_50.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    model = load_model("model/lstm/lstm_model_50.h5")
    seqs = tokenizer.texts_to_sequences(texts)
    pad = pad_sequences(seqs, maxlen=SEQ_LEN, padding="post")
    probs = model.predict(pad, verbose=0)
    label_col = "airline_sentiment" if "airline_sentiment" in df.columns else "sentiment"
    y_true = le.transform(df[label_col].str.lower())
    y_pred = np.argmax(probs, axis=1)
    evaluate(y_true, y_pred, label_names, "lstm")

if __name__ == "__main__":
    print("📥 Loading label encoder and test data...")
    le = load_label_encoder()
    label_names = le.classes_
    texts, df = load_and_prepare_test(TEST_PATH)

    test_bert(le, label_names, texts, df)
    test_lstm(le, label_names, texts, df)

    print("\n✅ All model evaluations complete! Check the model folders for plots and metrics.")
