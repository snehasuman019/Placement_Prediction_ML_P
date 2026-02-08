import streamlit as st
import requests

st.set_page_config(page_title="Placement Predictor", layout="centered")

st.title("🎓 AI Placement Prediction System")
st.write("Enter student details to predict placement chances")

# Inputs
cgpa = st.slider("CGPA", 0.0, 10.0, 8.0)
internships = st.number_input("Internships", 0, 10, 1)
projects = st.number_input("Projects", 0, 10, 2)
skills = st.number_input("Skills Score", 0, 10, 5)
communication = st.number_input("Communication Score", 0, 10, 6)

# Button
if st.button("Predict Placement"):

    data = {
        "CGPA": cgpa,
        "Internships": internships,
        "Projects": projects,
        "Skills": skills,
        "Communication": communication
    }

    try:
        response = requests.post(
            "http://localhost:8000/predict",
            json=data
        )

        result = response.json()

        st.success(f"Prediction: {result['prediction']}")
        st.info(f"Confidence: {round(result['confidence'] * 100, 2)}%")

    except:
        st.error("API not running! Start FastAPI server first.")
