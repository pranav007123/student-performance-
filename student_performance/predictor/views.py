import joblib
import os
from django.shortcuts import render

# Define model path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, 'ml_model', 'student_performance_model.pkl')

# Load the model
model = joblib.load(model_path)

def predict_student_performance(request):
    prediction = None  # Default value for rendering
    
    if request.method == "POST":
        # Get input values from the form
        study_hours = float(request.POST.get("study_hours", 0))
        attendance = float(request.POST.get("attendance", 0))
        previous_score = float(request.POST.get("previous_score", 0))

        # Make prediction
        features = [[study_hours, attendance, previous_score]]
        prediction = model.predict(features)[0]  # Extract single value
    
    return render(request, "predictor/index.html", {"prediction": prediction})
