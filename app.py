from flask import Flask, render_template, request
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load pre-trained model and scaler
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            # Get user inputs from the form
            pregnancies = float(request.form["pregnancies"])
            glucose = float(request.form["glucose"])
            blood_pressure = float(request.form["blood_pressure"])
            skin_thickness = float(request.form["skin_thickness"])
            insulin = float(request.form["insulin"])
            bmi = float(request.form["bmi"])
            diabetes_pedigree = float(request.form["diabetes_pedigree"])
            age = float(request.form["age"])

            # Validate the input values are within the specified range
            if not (0 <= pregnancies <= 17):
                raise ValueError("Pregnancies must be between 0 and 17.")
            if not (0 <= glucose <= 199):
                raise ValueError("Glucose must be between 0 and 199.")
            if not (0 <= blood_pressure <= 122):
                raise ValueError("Blood Pressure must be between 0 and 122.")
            if not (0 <= skin_thickness <= 99):
                raise ValueError("Skin Thickness must be between 0 and 99.")
            if not (0 <= insulin <= 846):
                raise ValueError("Insulin must be between 0 and 846.")
            if not (0 <= bmi <= 67):
                raise ValueError("BMI must be between 0 and 67.")
            if not (0.078 <= diabetes_pedigree <= 2.42):
                raise ValueError("Diabetes Pedigree Function must be between 0.078 and 2.42.")
            if not (21 <= age <= 81):
                raise ValueError("Age must be between 21 and 81.")

            # Prepare user input for prediction
            user_input = np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]).reshape(1, -1)
            user_input_scaled = scaler.transform(user_input)  # Scale the input data
            
            # Make prediction
            prediction = model.predict(user_input_scaled)
            prediction_proba = model.predict_proba(user_input_scaled)

            # Return prediction result to the user
            result = "Diabetes" if prediction[0] == 1 else "No Diabetes"
            probability = prediction_proba[0][1] * 100

            return render_template("index.html", result=result, probability=probability, message="Prediction made successfully!")
        
        except ValueError as e:
            return render_template("index.html", message=str(e))
    
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
