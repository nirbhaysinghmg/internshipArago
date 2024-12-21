# Video Link Of The Implementation and running the Code 

https://drive.google.com/file/d/1xd3gZTbTPhlkUy46mvCCCYnj1edkgj9E/view?usp=drive_link

# Shipment Delay Prediction Project

This project is made to predict if a shipment will be delayed or delivered on time. I trained a machine learning model and created an API to check predictions.

## How I Trained the Model

1.  **Data Preparation**:

    - First, I cleaned the data to make it ready for training.
    - I removed unnecessary columns like "Shipment ID" and the target column "Delayed" from the features.
    - Then, I transformed the data to have numerical values using one-hot encoding for categorical columns like "Origin," "Destination," "Weather Conditions," and "Traffic Conditions."

2.  **Model Selection**:

    - I used the `Logistic Regression` model from scikit-learn because it is simple and works well for classification tasks.
    - I also used the Random Forest but Logistic Regression was showing more accuracy so we selected it as the final model
    - After training, we saved the model using `pickle` so that we can reuse it in the API.

    ```python
    import pickle
    from sklearn.linear_model import LogisticRegression

    # Train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Save the model
    with open('finalized_model.sav', 'wb') as model_file:
        pickle.dump(model, model_file)

    ```

3.  **How i created the API**:

    - I have used flask to create the API in the Spyder
    - After writing the Flask code, we ran it locally using Spyder.
      -The API was hosted on http://127.0.0.1:5000/

4. **Creating API Endpoint**
    - I created a /predict endpoint that takes a JSON request with shipment details and returns whether the shipment is delayed or on time.
      The features are sent in the correct order as a list, and the model predicts the output.

      **THE CODE FOR THIS**
            from flask import Flask, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# Load the model
with open('finalized_model.sav', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({'prediction': 'Delayed' if prediction[0] == 1 else 'On Time'}), 200

if __name__ == '__main__':
    app.run(debug=True)

6.  **How i tested the API**: - Open Postman. - Create a POST request to http://127.0.0.1:5000/predict. - Send the shipment features in the body as JSON. Example:

        {
        "features": [400, 15, 6, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]

    }


    All the features in 0 and 1 are one hot encoded that is why there are a lot of features
    The total features are 30

# The API predicted the shipment if it will be delayed or not

# I have also attached a google drive link to the video of its working
