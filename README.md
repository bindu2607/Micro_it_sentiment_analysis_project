# ✈️ Twitter Airline Sentiment Analysis

**Full-Stack Machine Learning Project | NLP | BERT | BiLSTM | Streamlit**

A full-stack deep learning project for sentiment classification of tweets directed at U.S. airlines, using advanced NLP models like **BERT** and **BiLSTM**, wrapped in an interactive **Streamlit** web app.

## 📥 Model Files

Due to GitHub file size limits, large model files are hosted separately:

- **BERT Model**: [Download tf_model.h5](https://drive.google.com/file/d/YOUR_FILE_ID)
- **LSTM Model**: [Download lstm_model_50.h5](https://drive.google.com/file/d/YOUR_FILE_ID)
- **LSTM Tokenizer**: [Download tokenizer_lstm_50.pkl](https://drive.google.com/file/d/YOUR_FILE_ID)

### Setup with Models:
1. Download the model files from the links above
2. Place them in their respective folders as shown in the project structure
3. Run `streamlit run app.py`

---

## 🚀 Key Features

- 📊 **Exploratory Data Analysis (EDA)**  
  Visualize sentiment distribution, tweet length, and word clouds.

- 🧹 **Text Preprocessing**  
  - Handles emojis and emoticons  
  - Cleans mentions, hashtags, HTML, URLs  
  - Converts text to lowercase and handles negation

- 🧠 **Advanced Modeling**
  - **BERT**: Fine-tuned using HuggingFace Transformers (98.45% accuracy)
  - **BiLSTM**: Custom deep RNN with class balancing and padded sequences (97.36% accuracy)

- 📈 **Comprehensive Evaluation**
  - Precision, Recall, F1-Score reports  
  - Confusion matrices and accuracy plots  
  - Per-class performance metrics with visualizations

- 💻 **Interactive Deployment**
  - Built with **Streamlit**  
  - Single text analysis and batch CSV processing
  - Real-time sentiment prediction with confidence scores

- 💾 **Reproducible Artifacts**
  - Saved models and tokenizers
  - All evaluation plots and metrics in organized `model/` directory

---

## 🧪 Final Model Performance

| Model      | Accuracy | Precision | Recall   | F1-score |
|------------|----------|-----------|----------|----------|
| **BERT**   | **98.45%** | **98.45%** | **98.45%** | **98.45%** |
| **BiLSTM** | **97.36%** | **97.53%** | **97.36%** | **97.39%** |

---

📚 References

🤗 HuggingFace Transformers: https://huggingface.co/docs/transformers

📘 TensorFlow Keras: https://www.tensorflow.org/guide/keras

📊 Scikit-learn: https://scikit-learn.org/

😊 Emoji Package: https://pypi.org/project/emoji/


---

## ⚙️ Setup and Usage

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
'''


## Setup and Usage

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Train the model (optional if pre-trained provided):
   ```
   python train.py
   ```
3. Run the Streamlit app:
   ```
   streamlit run app.py
   
   The app will be available at `http://localhost:8501`
   ```
   
4.2. Test the model 
   ```
   python test.py
   ```
## 👩‍💻Author

Marpini Himabindu
B.Tech in Information Technology, 2022–2026

