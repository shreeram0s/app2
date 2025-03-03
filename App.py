import os
import requests
import docx2txt
import pdfminer.high_level
import PyPDF2
from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer, util
from googleapiclient.discovery import build

app = Flask(__name__)

# Udemy API Details (Replace with actual API Key)
UDEMY_API_URL = "https://www.udemy.com/api-2.0/courses/"
UDEMY_API_KEY = "YOUR_UDEMY_API_KEY"  # Replace with actual API key

# Google Search API Details (Backup Option)
GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY"  # Replace with actual key
SEARCH_ENGINE_ID = "YOUR_SEARCH_ENGINE_ID"  # Replace with actual ID

# Load Pre-trained Sentence Transformer Model
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")


def extract_text(file):
    """Extract text from PDF, DOCX, or TXT."""
    file_ext = file.filename.split(".")[-1].lower()

    if file_ext == "pdf":
        return pdfminer.high_level.extract_text(file)
    elif file_ext == "docx":
        return docx2txt.process(file)
    elif file_ext == "txt":
        return file.read().decode("utf-8")
    else:
        return ""


def extract_skills(text):
    """Extract skills from resume using sentence transformers."""
    predefined_skills = [
        "Python", "SQL", "Machine Learning", "Power BI", "Data Science",
        "AWS", "Java", "JavaScript", "React"
    ]

    extracted_skills = []
    for skill in predefined_skills:
        similarity = util.pytorch_cos_sim(model.encode(text), model.encode(skill))
        if similarity.item() > 0.5:  # Threshold for skill detection
            extracted_skills.append(skill)

    return extracted_skills


def fetch_udemy_courses(skill):
    """Fetch top courses from Udemy for a given skill."""
    headers = {"Authorization": f"Bearer {UDEMY_API_KEY}"}
    params = {"search": skill, "page_size": 5}

    response = requests.get(UDEMY_API_URL, headers=headers, params=params)

    if response.status_code == 200:
        courses = response.json().get("results", [])
        return [(course["title"], course["url"]) for course in courses]
    else:
        return fetch_google_courses(skill)  # Fallback to Google if Udemy fails


def fetch_google_courses(skill):
    """Fetch top courses from Google Search if Udemy API fails."""
    service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
    query = f"best {skill} online course site:udemy.com OR site:coursera.org OR site:edx.org"

    response = service.cse().list(q=query, cx=SEARCH_ENGINE_ID, num=5).execute()
    results = response.get("items", [])

    return [(item["title"], item["link"]) for item in results]


@app.route("/", methods=["GET", "POST"])
def home():
    missing_skills = []
    learning_schedule = {}

    if request.method == "POST":
        resume_file = request.files["resume"]
        job_desc_file = request.files["job_desc"]

        if resume_file and job_desc_file:
            resume_text = extract_text(resume_file)
            job_desc_text = extract_text(job_desc_file)

            resume_skills = set(extract_skills(resume_text))
            job_skills = set(extract_skills(job_desc_text))

            missing_skills = list(job_skills - resume_skills)  # Skills required but not in resume

            learning_schedule = {skill: fetch_udemy_courses(skill) for skill in missing_skills}

    return render_template("index.html", learning_schedule=learning_schedule, missing_skills=missing_skills)


if __name__ == "__main__":
    app.run(debug=True)
