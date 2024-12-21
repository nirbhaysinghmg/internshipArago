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

4.  **How i tested the API**: - Open Postman. - Create a POST request to http://127.0.0.1:5000/predict. - Send the shipment features in the body as JSON. Example:

        {
        "features": [400, 15, 6, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]

    }

# The API predicted the shipment if it will be delayed or not

# I have also attached a google drive link to the video of its working
