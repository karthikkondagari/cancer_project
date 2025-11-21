# # # # from flask import Flask, request, jsonify
# # # from flask_cors import CORS  # Import CORS
# # # import joblib
# # # import numpy as np
# # # from flask import Flask, request, jsonify


# # # # Initialize Flask app
# # # app = Flask(__name__)

# # # # Enable CORS for all routes
# # # CORS(app)

# # # # Load the trained Logistic Regression model and scaler
# # # model = joblib.load('logistic_regression_model.joblib')
# # # scaler = joblib.load('scaler.joblib')

# # # # Define a route for predictions
# # # @app.route('/predict', methods=['POST'])
# # # def predict():
# # #     try:
# # #         # Get data from the POST request
# # #         data = request.get_json(force=True)
        
# # #         # Extract features from the data
# # #         input_features = data['features']
        
# # #         # Convert the features to a numpy array and reshape for prediction
# # #         input_data = np.array(input_features).reshape(1, -1)
        
# # #         # Scale the input data using the loaded scaler
# # #         input_data_scaled = scaler.transform(input_data)
        
# # #         # Predict using the loaded model
# # #         prediction = model.predict(input_data_scaled)
        
# # #         # Convert numeric prediction to "Yes" or "No"
# # #         result = "Yes" if prediction[0] == 1 else "No"
        
# # #         # Return the result as a JSON response
# # #         return jsonify({'prediction': result})
    
# # #     except Exception as e:
# # #         return jsonify({'error': str(e)}), 400

# # # # Run the Flask app
# # # if __name__ == '__main__':
# # #     app.run(debug=True)



# # from flask import Flask, request, jsonify
# # import joblib
# # import numpy as np
# # from flask_cors import CORS  # For handling CORS (Cross-Origin Resource Sharing)

# # # Initialize Flask app
# # app = Flask(__name__)

# # # Enable CORS (allow cross-origin requests)
# # CORS(app)

# # # Load your pre-trained model and scaler
# # model = joblib.load('logistic_regression_model.joblib')  # Path to your trained model file
# # scaler = joblib.load('scaler.joblib')  # Path to your scaler file

# # # Route for prediction (only accepts POST requests)
# # @app.route('/predict', methods=['POST', 'GET'])
# # def predict():
# #     try:
# #         # Get data from the POST request (JSON format)
# #         data = request.get_json(force=True)

# #         # Extract features from the JSON data
# #         features = data['features']

# #         # Convert the list of features into a numpy array and reshape for prediction
# #         input_data = np.array(features).reshape(1, -1)

# #         # Scale the input data using the loaded scaler
# #         input_data_scaled = scaler.transform(input_data)

# #         # Make prediction using the trained model
# #         prediction = model.predict(input_data_scaled)

# #         # Return prediction as 'Yes' or 'No'
# #         result = "Yes" if prediction[0] == 1 else "No"

# #         return jsonify({'prediction': result})

# #     except Exception as e:
# #         return jsonify({'error': str(e)}), 400

# # # Run the Flask app
# # if __name__ == '__main__':
# #     app.run(debug=True)



# from flask import Flask, request, jsonify
# import joblib
# import numpy as np
# from flask_cors import CORS  # Import CORS

# # Initialize Flask app
# app = Flask(__name__)

# # Enable CORS (to allow requests from any origin)
# CORS(app)

# # Load pre-trained model and scaler
# model = joblib.load('logistic_regression_model.joblib')  # Replace with your model's file path
# scaler = joblib.load('scaler.joblib')  # Replace with your scaler's file path

# @app.route('/predict', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         try:
#             # Get JSON data from the request
#             data = request.get_json(force=True)

#             # Ensure the 'features' key exists
#             if 'features' not in data:
#                 return jsonify({'error': 'Missing "features" in request data'}), 400

#             # Extract features from the received data
#             features = data['features']

#             # Check if the number of features matches what the model expects (32 features in your case)
#             if len(features) != 32:
#                 return jsonify({'error': 'Incorrect number of features, expected 32'}), 400

#             # Prepare data for prediction (reshape and scale)
#             input_data = np.array(features).reshape(1, -1)
#             input_data_scaled = scaler.transform(input_data)

#             # Predict using the trained model
#             prediction = model.predict(input_data_scaled)

#             # Return the result as 'Yes' or 'No'
#             result = "Malignant" if prediction[0] == 1 else "Benign"
            
#             return jsonify({'prediction': result})

#         except Exception as e:
#             return jsonify({'error': str(e)}), 400

# # Run the Flask app
# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

# Load trained model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

app = Flask(__name__)
CORS(app)  # Allow requests from frontend

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input features from request
        data = request.json.get('features', [])

        # Validate input
        if not isinstance(data, list) or len(data) != 30:
            return jsonify({'error': 'Input must be a list of 30 numerical values.'}), 400

        # Scale input data
        data_scaled = scaler.transform([data])

        # Predict output (1 or 0)
        prediction = model.predict(data_scaled)[0]

        return jsonify({'prediction': int(prediction)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
