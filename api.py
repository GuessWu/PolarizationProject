from flask import Flask, request, jsonify,render_template
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
import os
from nltk.stem import WordNetLemmatizer
import joblib

app = Flask(__name__)

# Wczytanie wytrenowanego modelu i CountVectorizer
model = joblib.load("lr_predictions.pkl")
cv = joblib.load("count.pkl")
tfidf=joblib.load("tfidf.pkl")
nltk.download('stopwords')

@app.route('/')
def home():
    return render_template('text_article.html')

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
    text_vector = tfidf.transform([preprocessed_text])
    prediction = model.predict_proba(text_vector)
    return prediction[0]

# Endpoint dla przetwarzania danych
@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        input_text = request.form['text']
        prediction = predict_label(input_text)
        return render_template('return_bias.html', prediction=prediction)
    elif request.method == 'GET':
        return render_template('text_article.html', prediction=None)

@app.route('/health', methods=['GET'])
def health():    
	return jsonify({
    	'response' : True,
	})


if __name__ == '__main__':
	port = int(os.environ.get("PORT", 5000))
	app.run(debug=True, host='::', port=port)

