from flask import Flask, request, jsonify
import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import re
from nltk.corpus import stopwords
import os
from nltk.stem import WordNetLemmatizer
import joblib

app = Flask(__name__)

# Wczytanie wytrenowanego modelu i CountVectorizer
model = joblib.load("lr_predictions.pkl")
cv = joblib.load("count_vectorizer.pkl")

# Funkcja do przetwarzania tekstu
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [word for word in text if word not in stopwords.words('english')]
    text = ' '.join(text)
    return text

# Funkcja do przewidywania etykiety dla tekstu
def predict_label(text):
    preprocessed_text = preprocess_text(text)
    text_vector = cv.transform([preprocessed_text])
    prediction = model.predict(text_vector)
    return prediction[0]

# Endpoint dla przetwarzania danych
@app.route('/predict', methods=['GET','POST'])
def predict():
    data = request.get_json()
    input_text = data['text']
    prediction = predict_label(input_text)
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
	port = int(os.environ.get("PORT", 5000))
	app.run(debug=True, host='::', port=port)
