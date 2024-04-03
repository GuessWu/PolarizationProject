import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flasgger import Swagger

app = Flask(__name__)



@app.route('/predict', methods=['POST'])
def predict():

            json_ = request.json
            print(json_)
            query = pd.get_dummies(pd.DataFrame(json_))

            prediction = list(lr.predict(query))

            return jsonify({'prediction': str(prediction)})

if __name__ == '__main__':
    pipeline_filename = os.path.join('trained_model', 'lr_predictions.pkl')
    lr = joblib.load(pipeline_filename) # Load "model.pkl"
    print ('Model loaded')

    app.run(debug=True)