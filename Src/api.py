from fastapi import FastAPI
import pandas as pd
import joblib
import os

app = FastAPI(title="Placement Prediction API")

# Load ML artifacts

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = joblib.load(os.path.join(BASE_DIR, "placement_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
FEATURE_COLUMNS = joblib.load(os.path.join(BASE_DIR, "feature_columns.pkl"))

# Prediction function

def predict(student_data: dict):

    df = pd.DataFrame(columns=FEATURE_COLUMNS)

    for key, value in student_data.items():
        if key in df.columns:
            df.loc[0, key] = value

    df = df.fillna(0)
    df_scaled = scaler.transform(df)

    pred = model.predict(df_scaled)[0]
    prob = model.predict_proba(df_scaled)[0][1]

    return pred, prob

# API route

@app.post("/predict")
def predict_api(student: dict):

    result, probability = predict(student)

    return {
    "prediction": result,
    "confidence": round(float(probability), 3)
}
@app.get("/")
def home():
    return {"message": "Placement Prediction API is running 🚀"}

#input → API → preprocessing → scaler → model → JSON output