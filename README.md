# Spam-Detector
# 📧 Spam Detection System

An **Email & SMS Spam Detector** that uses **Natural Language Processing (NLP)** and **Machine Learning** to classify messages as **Spam** or **Ham (Not Spam)**.  
The app features an AI-assisted UI built with **Streamlit**, and uses **ChatGPT** to design, refine, and document the user interface and project workflow.

---

## 🚀 Features

✅ **AI-assisted UI** — designed and optimized with ChatGPT  
✅ **WordNet Lemmatization + POS Tagging** (NLTK) for text cleaning  
✅ **Random Forest Classifier** for robust spam detection  
✅ Achieves up to **98% accuracy** on test data  
✅ **Instant spam prediction** through an interactive web UI  
✅ Easy one-click retraining option  
✅ Clean, professional layout ready for portfolio/demo use  

---

## 🧠 How It Works

1. **Data Preprocessing**
   - Converts all text to lowercase  
   - Removes punctuation and stopwords  
   - Lemmatizes each word (using correct part-of-speech tag)

2. **Feature Extraction**
   - Converts text into numeric vectors using `CountVectorizer`

3. **Model Training**
   - Trains a `RandomForestClassifier` on the labeled SMS spam dataset  
   - Saves trained model and vectorizer with `joblib`

4. **Prediction & UI**
   - Loads the saved `.joblib` files  
   - Accepts user text input  
   - Displays real-time classification with probability  
   - Built entirely in **Streamlit** for a responsive web interface  

---
## 🎥 Live Preview

![App Demo](assets/demo.gif)
