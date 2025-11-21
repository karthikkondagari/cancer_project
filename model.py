# # 
# import joblib
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split

# def train_and_save_model(data_path, model_path):
#     # Load data
#     data = pd.read_csv(data_path)

#     # Preprocessing (if any)
#     # ...

#     # Separate features (X) and target (y)
#     X = data.drop('diagnosis', axis=1)  # Assuming 'diagnosis' is the target column
#     y = data['diagnosis']

#     # Scale the features
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)

#     # Split data
#     X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#     # Train model
#     lr = LogisticRegression()
#     lr.fit(X_train, y_train)

#     # Save model and scaler
#     model_and_scaler = {'model': lr, 'scaler': scaler}
#     joblib.dump(model_and_scaler, model_path)

# # Example usage
# data_path = r'C:\Users\konda\Desktop\cancer\data.csv'
# model_path = 'breast_cancer_model.pkl'
# train_and_save_model(data_path, model_path)
# Import necessary libraries
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib  # Import joblib for saving and loading models

# Load data
data = pd.read_csv(r'C:\Users\konda\Desktop\cancer\data.csv')

# Data preprocessing
data.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)

# Convert diagnosis column to numeric (M = 1, B = 0)
data['diagnosis'] = [1 if value == 'M' else 0 for value in data['diagnosis']]
data['diagnosis'] = data['diagnosis'].astype("category", copy=False)

# Define target and feature variables
y = data['diagnosis']
X = data.drop(['diagnosis'], axis=1)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Predict the test set
y_pred = lr.predict(X_test)

# Print performance metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save the trained model and scaler using joblib
joblib.dump(lr, 'logistic_regression_model.joblib')  # Save model
joblib.dump(scaler, 'scaler.joblib')  # Save the scaler

# Load the model and scaler from files (to test reloading functionality)
loaded_lr = joblib.load('logistic_regression_model.joblib')  # Load model
loaded_scaler = joblib.load('scaler.joblib')  # Load scaler

# Example of using the loaded model for prediction
X_test_scaled = loaded_scaler.transform(X_test)  # Don't forget to scale the features
y_pred_loaded = loaded_lr.predict(X_test_scaled)

# Print the accuracy of the loaded model
print("Accuracy of loaded model:", accuracy_score(y_test, y_pred_loaded))
