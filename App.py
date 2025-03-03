import streamlit as st
import requests
import pandas as pd
import pdfplumber
import docx2txt
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
import googleapiclient.discovery

# Load AI Model
st_model = SentenceTransformer('all-MiniLM-L6-v2')

# YouTube API Key (Replace with a new secured key)
YOUTUBE_API_KEY = "AIzaSyBoRgw0WE_KzTVNUvH8d4MiTo1zZ2SqKPI"
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

# Initialize session state
if "skills_analyzed" not in st.session_state:
    st.session_state.skills_analyzed = False
    st.session_state.missing_skills = []
    st.session_state.matching_score = 0.0

# Function to fetch courses from YouTube
def fetch_youtube_courses(skill):
    youtube = googleapiclient.discovery.build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=YOUTUBE_API_KEY)
    request = youtube.search().list(q=f"{skill} course", part="snippet", maxResults=5, type="video")
    response = request.execute()
    
    return [
        {"Title": item["snippet"]["title"], "Channel": item["snippet"]["channelTitle"], "Video Link": f'https://www.youtube.com/watch?v={item["id"]["videoId"]}'}
        for item in response["items"]
    ]

# Function to extract text from files
def extract_text(uploaded_file):
    if uploaded_file is not None:
        ext = uploaded_file.name.split(".")[-1].lower()
        if ext == "pdf":
            with pdfplumber.open(uploaded_file) as pdf:
                return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        elif ext in ["docx", "doc"]:
            return docx2txt.process(uploaded_file)
        elif ext == "txt":
            return uploaded_file.read().decode("utf-8")
        else:
            st.error("Unsupported file format! Please upload PDF, DOCX, or TXT.")
    return ""

# Function to generate short descriptions
def generate_summary(text):
    sentences = text.split(". ")[:3]
    return "... ".join(sentences) + "..." if sentences else "No content extracted."

# Function to extract skills
def extract_skills(text):
    skills = ["Python", "Machine Learning", "Data Science", "AI", "Deep Learning", "NLP", "SQL", "Power BI", "Tableau", "TensorFlow", "Pandas", "Numpy"]
    return [skill for skill in skills if skill.lower() in text.lower()]

# Function to calculate similarity score
def calculate_matching_score(resume_text, job_text):
    embeddings = st_model.encode([resume_text, job_text], convert_to_tensor=True)
    return round(float(util.pytorch_cos_sim(embeddings[0], embeddings[1])[0]), 2) * 100

# Function to plot skill comparison
def plot_skill_comparison(resume_skills, job_skills):
    all_skills = list(set(resume_skills + job_skills))
    resume_counts = [1 if skill in resume_skills else 0 for skill in all_skills]
    job_counts = [1 if skill in job_skills else 0 for skill in all_skills]
    
    df = pd.DataFrame({"Skills": all_skills, "Resume": resume_counts, "Job Requirements": job_counts})
    df.set_index("Skills").plot(kind="bar", figsize=(8, 4), color=["blue", "red"], alpha=0.7)
    plt.title("Resume vs. Job Skills Comparison")
    plt.xticks(rotation=45)
    plt.ylabel("Presence (1 = Present, 0 = Missing)")
    st.pyplot(plt)

# Streamlit UI
st.title("📄 AI Resume Analyzer & Skill Enhancer")
st.write("Upload your Resume and Job Description to analyze missing skills and get YouTube course recommendations!")

# File Uploaders
resume_file = st.file_uploader("📄 Upload Resume (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
job_file = st.file_uploader("📄 Upload Job Description (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

if resume_file and job_file:
    resume_text = extract_text(resume_file)
    job_text = extract_text(job_file)
    
    st.subheader("📌 Resume Summary")
    st.write(generate_summary(resume_text))
    
    st.subheader("📌 Job Description Summary")
    st.write(generate_summary(job_text))
    
    if st.button("Analyze Skills & Matching Score"):
        resume_skills = extract_skills(resume_text)
        job_skills = extract_skills(job_text)
        missing_skills = list(set(job_skills) - set(resume_skills))
        
        st.session_state.skills_analyzed = True
        st.session_state.missing_skills = missing_skills
        st.session_state.matching_score = calculate_matching_score(resume_text, job_text)
        
        st.subheader("🔍 Extracted Skills")
        st.write(f"**Resume Skills:** {', '.join(resume_skills)}")
        st.write(f"**Job Required Skills:** {', '.join(job_skills)}")
        
        st.subheader("📊 Resume Matching Score")
        st.write(f"Your resume matches **{st.session_state.matching_score}%** of the job requirements.")
        
        st.subheader("⚠️ Missing Skills")
        if missing_skills:
            st.write(f"You are missing: {', '.join(missing_skills)}")
        else:
            st.success("You have all the required skills!")
        
        plot_skill_comparison(resume_skills, job_skills)

if st.session_state.skills_analyzed and st.session_state.missing_skills:
    if st.button("📚 Get Recommended Courses"):
        all_courses = []
        for skill in st.session_state.missing_skills:
            all_courses.extend(fetch_youtube_courses(skill))
        df = pd.DataFrame(all_courses)
        st.table(df if not df.empty else "No courses found.")
