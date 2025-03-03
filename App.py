import streamlit as st
import requests
import pandas as pd
import PyPDF2
import pdfplumber
import docx2txt
import numpy as np
from sentence_transformers import SentenceTransformer

# Load AI Model
st_model = SentenceTransformer('all-MiniLM-L6-v2')

# API for fetching courses (Replace with actual API link)
COURSE_API_URL = "https://api.example.com/courses"

# Function to extract text from uploaded file
def extract_text_from_file(uploaded_file):
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split(".")[-1].lower()
        
        if file_extension == "pdf":
            with pdfplumber.open(uploaded_file) as pdf:
                text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        elif file_extension in ["docx", "doc"]:
            text = docx2txt.process(uploaded_file)
        elif file_extension == "txt":
            text = uploaded_file.read().decode("utf-8")
        else:
            st.error("Unsupported file format! Please upload a PDF, DOCX, or TXT file.")
            return None
        return text
    return None

# Function to extract skills from text
def extract_skills(text):
    skills_list = ["Python", "Machine Learning", "Data Science", "AI", "Deep Learning", "NLP", "SQL", "Power BI", "Tableau", "TensorFlow", "Pandas", "Numpy"]
    
    found_skills = [skill for skill in skills_list if skill.lower() in text.lower()]
    return found_skills

# Function to find missing skills
def find_missing_skills(resume_skills, job_skills):
    return list(set(job_skills) - set(resume_skills))

# Fetch online courses from API
def fetch_courses():
    try:
        response = requests.get(COURSE_API_URL)
        if response.status_code == 200:
            return response.json()  # Return JSON response
        else:
            st.error(f"Failed to fetch courses: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Streamlit UI
st.title("📄 AI Resume Analyzer & Skill Enhancer")
st.write("Upload your Resume and Job Description to analyze missing skills and get online courses!")

# File Uploaders
resume_file = st.file_uploader("📄 Upload Resume (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
job_file = st.file_uploader("📄 Upload Job Description (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

if st.button("Analyze Skills"):
    if resume_file and job_file:
        resume_text = extract_text_from_file(resume_file)
        job_text = extract_text_from_file(job_file)

        if resume_text and job_text:
            resume_skills = extract_skills(resume_text)
            job_skills = extract_skills(job_text)
            missing_skills = find_missing_skills(resume_skills, job_skills)

            st.subheader("🔍 Extracted Skills")
            st.write(f"**Resume Skills:** {', '.join(resume_skills)}")
            st.write(f"**Job Required Skills:** {', '.join(job_skills)}")

            st.subheader("⚠️ Missing Skills")
            if missing_skills:
                st.write(f"You are missing: {', '.join(missing_skills)}")
            else:
                st.success("You have all the required skills!")

if st.button("📚 Get Recommended Courses"):
    courses = fetch_courses()
    
    if courses:
        df = pd.DataFrame(courses)  # Convert JSON to DataFrame
        st.table(df)  # Display as table
    else:
        st.write("No courses found.")

# Run Streamlit with: streamlit run app.py
