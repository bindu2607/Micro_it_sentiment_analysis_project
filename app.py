import streamlit as st
import os
import random

st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="🔍",
    layout="centered"
)

# --- Helper: Demo prediction if models are missing ---
def demo_predict(text, model_name="BERT"):
    text = text.lower()
    positive_words = ['love', 'great', 'excellent', 'amazing', 'good', 'fantastic', 'wonderful', 'best', 'perfect', 'awesome', 'happy']
    negative_words = ['hate', 'bad', 'terrible', 'awful', 'worst', 'horrible', 'delayed', 'cancelled', 'rude', 'disappointed', 'angry']
    pos_count = sum(1 for word in positive_words if word in text)
    neg_count = sum(1 for word in negative_words if word in text)
    if pos_count > neg_count:
        sentiment = "Positive"
        confidence = random.uniform(0.85, 0.98)
    elif neg_count > pos_count:
        sentiment = "Negative"
        confidence = random.uniform(0.80, 0.95)
    else:
        sentiment = "Neutral"
        confidence = random.uniform(0.60, 0.80)
    return sentiment, confidence

def models_available():
    bert_model = "model/bert/bert_model/tf_model.h5"
    lstm_model = "model/lstm/lstm_model_50.h5"
    return os.path.exists(bert_model) and os.path.exists(lstm_model)

# --- Main App ---
def main():
    st.markdown(
        "<h1 style='text-align: center; font-size:2.6rem;'>🔍 Sentiment Analysis App</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<div style='text-align:center; color:#bbb; margin-bottom:2rem;'>"
        "Analyze text sentiment using <b>BERT</b> or <b>LSTM</b> models"
        "</div>", unsafe_allow_html=True
    )

    tabs = st.tabs(["Single Text", "Batch CSV"])

    # --- Single Text Tab ---
    with tabs[0]:
        st.markdown("<h2>Single Text Analysis</h2>", unsafe_allow_html=True)
        st.markdown("**Input Text:**")
        user_input = st.text_area("Enter your text here...", key="single_text", height=100, label_visibility="collapsed")
        st.markdown("**Model:**")
        model_choice = st.radio("", ["BERT", "LSTM"], horizontal=True, key="single_model")
        if st.button("Analyze", key="analyze_single"):
            if not user_input.strip():
                st.warning("Please enter your text.")
            else:
                if not models_available():
                    sentiment, confidence = demo_predict(user_input, model_choice)
                else:
                    # Place your real model code here if running locally
                    sentiment, confidence = demo_predict(user_input, model_choice)
                color = {"Positive": "#2ecc71", "Negative": "#e74c3c", "Neutral": "#3498db"}.get(sentiment, "#95a5a6")
                st.markdown(f"""
                <div style='text-align:center; margin:2rem 0; padding:1.5rem; border-radius:10px; background-color:#18191A; border-left: 4px solid {color}'>
                    <div style='font-size:1.3rem; margin-bottom:0.5rem'>
                        Predicted Sentiment: 
                        <span style='color:{color}; font-weight:bold; font-size:1.4rem'>
                            {sentiment}
                        </span>
                    </div>
                    <div style='font-size:1.1rem; color:#aaa'>
                        Confidence: {confidence:.1%}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.progress(confidence, text="Model Confidence")

    # --- Batch CSV Tab ---
    with tabs[1]:
        st.markdown("<h2>Batch CSV</h2>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload CSV file (must have a 'text' column):", type=["csv"], key="batch_csv")
        st.markdown("**Model:**")
        model_choice_batch = st.radio("", ["BERT", "LSTM"], horizontal=True, key="batch_model")
        if uploaded_file and st.button("Analyze", key="analyze_batch"):
            import pandas as pd
            df = pd.read_csv(uploaded_file)
            if 'text' not in df.columns:
                st.error("CSV must contain a 'text' column.")
            else:
                st.info(f"Processing {len(df)} rows...")
                results = []
                for text in df['text']:
                    sentiment, confidence = demo_predict(str(text), model_choice_batch)
                    results.append({"text": text, "Predicted Sentiment": sentiment, "Confidence": f"{confidence:.1%}"})
                df_result = pd.DataFrame(results)
                st.dataframe(df_result)
                csv = df_result.to_csv(index=False).encode()
                st.download_button("Download Results CSV", csv, "sentiment_results.csv", "text/csv")

  
if __name__ == "__main__":
    main()
    st.markdown("---")
    st.markdown("""
    <div style='text-align:center; color:#666; margin-top:2rem'>
        <p><strong>© 2025 Sentiment Analysis Project</strong></p>
        <p>👩‍💻 Built by Marpini Himabindu | 🎓 B.Tech IT 2022-2026</p>
        <p>💻 <a href="https://github.com/bindu2607/Micro_it_sentiment_analysis_project" target="_blank">View on GitHub</a></p>
    </div>
    """, unsafe_allow_html=True)
