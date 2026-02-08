import pdfplumber
import spacy

nlp = spacy.load("en_core_web_sm")

# Basic skill dictionary
SKILLS = [
    "python", "java", "sql", "machine learning", "data science",
    "deep learning", "excel", "communication", "c++", "react",
    "node", "cloud", "statistics", "git", "ai"
]

def extract_text(pdf_path):
    text = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"

    return text.lower()


def extract_skills(text):

    found = set()

    for skill in SKILLS:
        if skill in text:
            found.add(skill)

    return list(found)


def parse_resume(pdf_path):

    text = extract_text(pdf_path)
    skills = extract_skills(text)

    return {
        "skills": skills,
        "skill_count": len(skills)
    }
