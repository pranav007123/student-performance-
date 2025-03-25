import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
import joblib

# Define dataset path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, 'dataset.csv')

# Load dataset
df = pd.read_csv(data_path)

# Handle missing values (if any)
df.fillna(df.mean(), inplace=True)

# Separate features (X) and target (y)
X = df[['StudyHours', 'Attendance', 'PreviousScores']]
y = df['FinalGrade']

# Apply Polynomial Features (Degree 2)
poly = PolynomialFeatures(degree=2, include_bias=False)

# Create a pipeline with scaling and regression
model = make_pipeline(poly, StandardScaler(), LinearRegression())

# Train the model
model.fit(X, y)

# Save the trained model
model_path = os.path.join(BASE_DIR, 'student_performance_model.pkl')
joblib.dump(model, model_path)

print("✅ Model training complete! The trained model is saved as 'student_performance_model.pkl'.")
