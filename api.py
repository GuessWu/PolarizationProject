import os
import joblib
import pandas as pd
from flask import Flask, request
from flasgger import Swagger

app = Flask(__name__)
Swagger(app)

pipeline_filename = os.path.join('trained_model', 'lr_predictions.pkl')
pipeline = joblib.load(pipeline_filename)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict political bias for new data
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          id: PredictInput
          properties:
            text:
              type: string
              description: Input article text
    responses:
      200:
        description: Successfully predicted political bias
        schema:
          id: PredictOutput
          properties:
            prediction:
              type: string
              description: Predicted political bias (left/right)
    """
    data = request.get_json()

    new_data = pd.DataFrame(data, index=[0])

    prediction = pipeline.predict(new_data)
    return f"Model prediction is {prediction}"

if __name__ == '__main__':
    app.run(debug=True)