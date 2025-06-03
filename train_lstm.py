# train_lstm.py

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, GlobalMaxPool1D, Dense, Dropout
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import pickle

from data_utils import load_and_prepare_train

# ============== CONFIG ==============
os.makedirs("model/lstm", exist_ok=True)

SEQ_LEN   = 50
VOCAB_SIZE= 20000
BATCH_SIZE= 64
EPOCHS    = 10

TRAIN_CSV = "data/train.csv"

# ============ BiLSTM TRAINING ============
def train_bilstm(X_train, y_train, num_classes):
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    X_seq = tokenizer.texts_to_sequences(X_train)
    X_pad = pad_sequences(X_seq, maxlen=SEQ_LEN, padding="post")

    class_weights = dict(
        enumerate(compute_class_weight("balanced", classes=np.unique(y_train), y=y_train))
    )

    model = Sequential([
        Embedding(VOCAB_SIZE, 128, input_length=SEQ_LEN),
        Bidirectional(LSTM(64, return_sequences=True, dropout=0.3)),
        Bidirectional(LSTM(32, return_sequences=True, dropout=0.3)),
        GlobalMaxPool1D(),
        Dense(64, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ])

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="loss", patience=2, restore_best_weights=True, verbose=1
    )

    history = model.fit(
        X_pad, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        verbose=2,
        callbacks=[early_stop]
    )

    # Save model + tokenizer
    model.save("model/lstm/lstm_model_50.h5")
    with open("model/lstm/tokenizer_lstm_50.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    # Plot training curves
    plt.figure(figsize=(6,4))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.title("BiLSTM Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("model/lstm/lstm_train_loss.png")
    plt.close()

    plt.figure(figsize=(6,4))
    plt.plot(history.history["accuracy"], label="Train Acc")
    plt.title("BiLSTM Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("model/lstm/lstm_train_acc.png")
    plt.close()

    return model, tokenizer

# ============== MAIN ==============
if __name__ == "__main__":
    print("🧠 Loading and preprocessing train.csv for BiLSTM...")
    X_train, y_train, le, num_classes = load_and_prepare_train(TRAIN_CSV)

    print("🧠 Training BiLSTM on train.csv...")
    lstm_model, lstm_tokenizer = train_bilstm(X_train, y_train, num_classes)
    print("✅ BiLSTM model and tokenizer saved under model/lstm/")
