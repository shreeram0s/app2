import streamlit as st
import pandas as pd
import pdfplumber
import re
import spacy
import requests
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load NLP Model
nlp = spacy.load("en_core_web_sm")
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")  # Efficient for similarity tasks

# Extract text from PDF
def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + " "
    return text

# Extract skills using Named Entity Recognition (NER) and Regex
def extract_skills(text):
    skills = set()
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PRODUCT", "WORK_OF_ART"]:  # Skills often detected as these entities
            skills.add(ent.text)
    
    # Additional regex-based skill extraction
    predefined_skills = {"HTML", "CSS", "JavaScript", "Python", "React", "Django", "Node.js", "SQL", "MongoDB"}
    skills.update({skill for skill in predefined_skills if re.search(rf"\b{skill}\b", text, re.IGNORECASE)})
    
    return skills

# Find missing skills
def find_missing_skills(resume_text, job_desc_text):
    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_desc_text)
    missing_skills = job_skills - resume_skills
    return list(missing_skills), list(resume_skills), list(job_skills)

# Fetch learning resources dynamically
def fetch_learning_resources(skill):
    search_query = f"{skill} online course"
    url = f"https://www.googleapis.com/customsearch/v1?q={search_query}&key=YOUR_GOOGLE_API_KEY&cx=YOUR_SEARCH_ENGINE_ID"
    
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        courses = [(item["title"], item["link"]) for item in data.get("items", [])[:5]]
        return courses
    return []

# Generate a structured learning plan
def generate_learning_plan(missing_skills):
    schedule = []
    for skill in missing_skills:
        resources = fetch_learning_resources(skill)
        for day, (title, link) in enumerate(resources, 1):
            schedule.append((f"Day {day}", skill, title, link))
    return pd.DataFrame(schedule, columns=["Day", "Skill", "Course Title", "Resource Link"])

# Streamlit UI
st.title("📄 AI Resume Analyzer - Skill Gap Learning Plan")
st.subheader("📌 Upload your resume and job description to identify skill gaps!")

# Upload resume
resume_file = st.file_uploader("Upload your Resume (PDF)", type=["pdf"])
job_desc = st.text_area("Paste the Job Description")

if resume_file and job_desc:
    with st.spinner("Processing..."):
        resume_text = extract_text_from_pdf(resume_file)
        missing_skills, resume_skills, job_skills = find_missing_skills(resume_text, job_desc)

        # Display Resume and Job Description summaries
        st.subheader("🔎 Resume Summary")
        st.write(resume_text[:500] + "...")

        st.subheader("📄 Job Description Overview")
        st.write(job_desc[:500] + "...")

        # Skill Matching Visualization
        st.subheader("📊 Skill Matching Visualization")
        skill_labels = ["Matched Skills", "Missing Skills"]
        skill_counts = [len(resume_skills), len(missing_skills)]

        fig, ax = plt.subplots()
        ax.bar(skill_labels, skill_counts, color=["green", "red"])
        ax.set_ylabel("Count")
        ax.set_title("Resume vs. Job Description Skills")
        st.pyplot(fig)

        if missing_skills:
            st.success(f"🔍 Missing Skills Identified: {', '.join(missing_skills)}")

            # Recommendations and Learning Plan
            st.subheader("📌 Recommendations")
            st.write("Learn these skills using the suggested online courses below.")

            schedule_df = generate_learning_plan(missing_skills)
            st.subheader("📅 Personalized Learning Schedule")
            st.dataframe(schedule_df)
        else:
            st.success("✅ No missing skills detected! Your resume is well-matched.")
