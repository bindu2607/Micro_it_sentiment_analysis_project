# train_bert.py

import os
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification, create_optimizer
import matplotlib.pyplot as plt
from data_utils import load_and_prepare_train

# ============== CONFIG ==============
os.makedirs("model/bert", exist_ok=True)

BERT_MAX_LEN  = 128
BATCH_SIZE    = 16
EPOCHS        = 4
LEARNING_RATE = 3e-5

TRAIN_CSV = "data/train.csv"

# ============ BERT TRAINING ============
def train_bert(X_train, y_train, num_classes):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    train_enc = tokenizer(
        X_train,
        truncation=True,
        padding="max_length",
        max_length=BERT_MAX_LEN,
        return_tensors="tf"
    )

    train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_enc), y_train)) \
                        .shuffle(1000).batch(BATCH_SIZE)

    model = TFBertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=num_classes
    )

    steps = int(len(X_train) / BATCH_SIZE) * EPOCHS
    optimizer, _ = create_optimizer(
        init_lr=LEARNING_RATE, num_train_steps=steps, num_warmup_steps=0
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="loss", patience=1, restore_best_weights=True, verbose=1
    )

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )

    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        callbacks=[early_stop]
    )

    # Save model and tokenizer
    model.save_pretrained("model/bert/bert_model")
    tokenizer.save_pretrained("model/bert/bert_tokenizer")

    # Plot training curves
    plt.figure(figsize=(6,4))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.title("BERT Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("model/bert/bert_train_loss.png")
    plt.close()

    plt.figure(figsize=(6,4))
    plt.plot(history.history["accuracy"], label="Train Acc")
    plt.title("BERT Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("model/bert/bert_train_acc.png")
    plt.close()

    return model, tokenizer

# ============== MAIN ==============
if __name__ == "__main__":
    print("🧠 Loading and preprocessing train.csv for BERT...")
    X_train, y_train, le, num_classes = load_and_prepare_train(TRAIN_CSV)

    print("🧠 Training BERT on train.csv...")
    bert_model, bert_tokenizer = train_bert(X_train, y_train, num_classes)
    print("✅ BERT model and tokenizer saved under model/bert/")
