from fastapi import FastAPI
import pandas as pd
import joblib
import os
from Src.nlp.resume_parser import parse_resume
# from nlp.resume_parser import parse_resume

app = FastAPI(title="Placement Prediction API")

# Load ML artifacts

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = joblib.load(os.path.join(BASE_DIR, "placement_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
FEATURE_COLUMNS = joblib.load(os.path.join(BASE_DIR, "feature_columns.pkl"))

# Prediction function

def predict(student_data: dict):

    # Create row with all features initialized to 0
    df = pd.DataFrame([ {col: 0 for col in FEATURE_COLUMNS} ])

    # Overwrite provided values
    for key, value in student_data.items():
        if key in df.columns:
            df.at[0, key] = value

    df_scaled = scaler.transform(df)

    pred = model.predict(df_scaled)[0]
    prob = model.predict_proba(df_scaled)[0][1]

    return pred, prob



from fastapi import UploadFile, File

@app.post("/resume_predict")
async def resume_predict(file: UploadFile = File(...)):

    # Save uploaded resume temporarily
    temp_path = f"temp_{file.filename}"

    with open(temp_path, "wb") as f:
        f.write(await file.read())

    # Parse resume
    parsed = parse_resume(temp_path)

    # Convert NLP output → ML features
    student_data = {
        "Skills": parsed["skill_count"]
    }

    result, probability = predict(student_data)

    # Cleanup temp file
    os.remove(temp_path)

    return {
        "skills_found": parsed["skills"],
        "prediction": result,
        "confidence": round(float(probability), 3)
    }


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