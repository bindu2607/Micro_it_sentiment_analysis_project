# --------- STREAMLIT CONFIG MUST BE FIRST ---------
import streamlit as st
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="🔍",
    layout="centered"
)

# --------- IMPORT OTHER LIBRARIES ---------
import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import TFBertForSequenceClassification, BertTokenizer

# Suppress TensorFlow logging
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --------- CONFIG ---------
SEQ_LEN = 50
BERT_MAX_LEN = 128
MODEL_DIR = "model"
BERT_MODEL_PATH = os.path.join(MODEL_DIR, "bert", "bert_model")
BERT_TOKENIZER_PATH = os.path.join(MODEL_DIR, "bert", "bert_tokenizer")
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, "lstm", "lstm_model_50.h5")
LSTM_TOKENIZER_PATH = os.path.join(MODEL_DIR, "lstm", "tokenizer_lstm_50.pkl")

# --------- CLEANING FUNCTION ---------
def clean_text(text):
    import re, emoji
    text = str(text)
    text = emoji.demojize(text)
    text = text.lower()
    text = re.sub(r"http\S+|@\w+|#\w+|[^\w\s:]", "", text)
    text = re.sub(r":([a-z_]+):", r" \1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# --------- MODEL LOADING ---------
@st.cache_resource
def load_models():
    try:
        # Load label encoder
        df = pd.read_csv("data/train.csv")
        label_col = "airline_sentiment" if "airline_sentiment" in df.columns else "sentiment"
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        le.fit(df[label_col].str.lower())
        
        # Load BERT components
        bert_tokenizer = BertTokenizer.from_pretrained(BERT_TOKENIZER_PATH)
        bert_model = TFBertForSequenceClassification.from_pretrained(BERT_MODEL_PATH)
        
        # Load LSTM components
        with open(LSTM_TOKENIZER_PATH, "rb") as f:
            lstm_tokenizer = pickle.load(f)
        lstm_model = tf.keras.models.load_model(LSTM_MODEL_PATH)
        
        return {
            "bert": (bert_model, bert_tokenizer),
            "lstm": (lstm_model, lstm_tokenizer),
            "label_encoder": le,
            "labels": le.classes_
        }
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.stop()

# --------- PREDICTION FUNCTIONS ---------
def predict_bert(text, model, tokenizer, labels):
    try:
        text_clean = clean_text(text)
        enc = tokenizer([text_clean], 
                      truncation=True, 
                      padding="max_length", 
                      max_length=BERT_MAX_LEN, 
                      return_tensors="tf")
        logits = model(enc).logits
        probs = tf.nn.softmax(logits, axis=1).numpy()[0]
        idx = np.argmax(probs)
        return labels[idx].capitalize(), float(probs[idx])
    except Exception as e:
        st.error(f"BERT Prediction Error: {str(e)}")
        return "Error", 0.0

def predict_lstm(text, model, tokenizer, labels):
    try:
        text_clean = clean_text(text)
        seq = tokenizer.texts_to_sequences([text_clean])
        padded = pad_sequences(seq, maxlen=SEQ_LEN, padding='post')
        probs = model.predict(padded, verbose=0)
        idx = np.argmax(probs[0])
        return labels[idx].capitalize(), float(probs[0][idx])
    except Exception as e:
        st.error(f"LSTM Prediction Error: {str(e)}")
        return "Error", 0.0

# --------- STREAMLIT UI ---------
def main():
    st.title("🔍 Sentiment Analysis App")
    st.markdown("Analyze text sentiment using BERT or LSTM models")
    
    tab1, tab2 = st.tabs(["Single Text", "Batch CSV"])
    
    with tab1:
        st.subheader("Single Text Analysis")
        user_input = st.text_area("Input Text:", height=120, placeholder="Enter your text here...")
        model_choice = st.radio("Model:", ["BERT", "LSTM"], horizontal=True)
        
        if st.button("Analyze"):
            if not user_input.strip():
                st.warning("Please enter some text")
            else:
                with st.spinner("Analyzing..."):
                    try:
                        if model_choice == "BERT":
                            label, confidence = predict_bert(user_input, *models["bert"], models["labels"])
                        else:  # LSTM
                            label, confidence = predict_lstm(user_input, *models["lstm"], models["labels"])
                            
                        color = {"Positive": "green", "Negative": "red", "Neutral": "blue"}.get(label, "gray")
                        
                        st.markdown(f"""
                        <div style='text-align: center; margin: 2rem 0;'>
                            <div style='font-size: 1.4rem; margin-bottom: 0.5rem;'>
                                Predicted Sentiment: 
                                <span style='color: {color}; font-weight: bold; font-size: 1.6rem;'>
                                    {label}
                                </span>
                            </div>
                            <div style='font-size: 1.2rem;'>
                                Confidence: {confidence:.2%}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Confidence bar
                        st.markdown("**Confidence Level**")
                        st.progress(confidence)
                        
                    except Exception as e:
                        st.error(f"Analysis failed: {str(e)}")
    
    with tab2:
        st.subheader("Batch CSV Analysis")
        uploaded_file = st.file_uploader("Upload CSV:", type=["csv"], help="CSV must contain a 'text' column")
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                if 'text' not in df.columns:
                    st.error("CSV file must contain a 'text' column")
                else:
                    st.write(f"**File loaded:** {len(df)} rows")
                    st.dataframe(df.head(3))
                    
                    model_choice = st.radio("Select Model for Batch Processing:", 
                                           ["BERT", "LSTM"], 
                                           key="batch", 
                                           horizontal=True)
                    
                    if st.button("Process Batch"):
                        with st.spinner(f"Processing {len(df)} entries with {model_choice}..."):
                            try:
                                texts = df['text'].astype(str).tolist()
                                cleaned = [clean_text(t) for t in texts]
                                
                                if model_choice == "BERT":
                                    # Batch BERT prediction
                                    enc = models["bert"][1](cleaned, 
                                                          truncation=True, 
                                                          padding="max_length", 
                                                          max_length=BERT_MAX_LEN,
                                                          return_tensors="tf")
                                    logits = models["bert"][0](enc).logits
                                    probs = tf.nn.softmax(logits, axis=1).numpy()
                                else:
                                    # Batch LSTM prediction
                                    seqs = models["lstm"][1].texts_to_sequences(cleaned)
                                    padded = pad_sequences(seqs, maxlen=SEQ_LEN, padding='post')
                                    probs = models["lstm"][0].predict(padded, verbose=0)
                                
                                # Process results
                                predictions = [models["labels"][np.argmax(p)].capitalize() for p in probs]
                                confidences = np.max(probs, axis=1)
                                
                                df['Predicted_Sentiment'] = predictions
                                df['Confidence'] = confidences
                                
                                st.success("✅ Batch processing complete!")
                                
                                # Show results
                                st.subheader("Results Preview")
                                st.dataframe(df[['text', 'Predicted_Sentiment', 'Confidence']], height=400)
                                
                                # Summary stats
                                sentiment_counts = df['Predicted_Sentiment'].value_counts()
                                st.subheader("Sentiment Distribution")
                                st.bar_chart(sentiment_counts)
                                
                                # Download button
                                csv = df.to_csv(index=False).encode()
                                st.download_button(
                                    "📥 Download Results",
                                    data=csv,
                                    file_name=f"sentiment_results_{model_choice.lower()}.csv",
                                    mime="text/csv"
                                )
                                
                            except Exception as e:
                                st.error(f"Batch processing failed: {str(e)}")
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")

# --------- INITIALIZE APP ---------
if __name__ == "__main__":
    models = load_models()
    main()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>© 2025 Sentiment Analysis Project | Built with BERT & LSTM</p>
        <p><strong>BERT:</strong> 98.45% Accuracy | <strong>LSTM:</strong> 97.36% Accuracy</p>
    </div>
    """, unsafe_allow_html=True)
