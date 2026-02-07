""" Placement Prediction Inference Engine

This module loads a trained machine learning model and scaler,
accepts a student's profile as input, preprocesses it,
and returns a placement prediction with probability.

This file represents the production inference pipeline
for the AI Placement Prediction System.

Workflow:
1. Load saved model and scaler
2. Accept new student input
3. Convert input to DataFrame
4. Apply feature scaling
5. Predict placement outcome
"""


import pandas as pd
import joblib
import os
import warnings

warnings.filterwarnings("ignore")

# Locate project root

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "placement_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
FEATURE_PATH = os.path.join(BASE_DIR, "feature_columns.pkl")

# Load ML artifacts


model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
FEATURE_COLUMNS = joblib.load(FEATURE_PATH)

# Prediction function
def predict_placement(student_data: dict):

    # Create empty dataframe with training schema
    df = pd.DataFrame(columns=FEATURE_COLUMNS)

    # Insert provided student values
    for key, value in student_data.items():
        if key in df.columns:
            df.loc[0, key] = value

    # Fill missing features with 0
    df = df.fillna(0)

    # Scale features
    df_scaled = scaler.transform(df)

    # Predict
    prediction = model.predict(df_scaled)[0]
    probability = model.predict_proba(df_scaled)[0][1]

    return prediction, probability


# Example test run

if __name__ == "__main__":

    sample_student = {
        "CGPA": 8.2,
        "Internships": 2,
        "Projects": 3,
        "Skills": 5,
        "Communication": 7
    }

    result, prob = predict_placement(sample_student)

    print("\n=== Placement Prediction ===")
    print("Result:", "Placed" if result == 1 else "Not Placed")
    print("Confidence:", round(prob * 100, 2), "%")
