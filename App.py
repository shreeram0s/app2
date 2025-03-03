import streamlit as st
import requests
import pandas as pd
import PyPDF2
import pdfplumber
import docx2txt
import numpy as np
from sentence_transformers import SentenceTransformer
import googleapiclient.discovery

# Load AI Model
st_model = SentenceTransformer('all-MiniLM-L6-v2')

# YouTube API Key (Replace with a new secured key)
YOUTUBE_API_KEY = "AIzaSyBoRgw0WE_KzTVNUvH8d4MiTo1zZ2SqKPI"
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

# Initialize session state for tracking analysis
if "skills_analyzed" not in st.session_state:
    st.session_state.skills_analyzed = False
    st.session_state.missing_skills = []

# Function to fetch courses from YouTube
def fetch_youtube_courses(skill):
    youtube = googleapiclient.discovery.build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=YOUTUBE_API_KEY)
    request = youtube.search().list(
        q=f"{skill} course",
        part="snippet",
        maxResults=5,
        type="video"
    )
    response = request.execute()
    
    courses = []
    for item in response["items"]:
        courses.append({
            "Title": item["snippet"]["title"],
            "Channel": item["snippet"]["channelTitle"],
            "Video Link": f'https://www.youtube.com/watch?v={item["id"]["videoId"]}'
        })
    return courses

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
    return [skill for skill in skills_list if skill.lower() in text.lower()]

# Function to find missing skills
def find_missing_skills(resume_skills, job_skills):
    return list(set(job_skills) - set(resume_skills))

# Streamlit UI
st.title("📄 AI Resume Analyzer & Skill Enhancer")
st.write("Upload your Resume and Job Description to analyze missing skills and get YouTube course recommendations!")

# File Uploaders
resume_file = st.file_uploader("📄 Upload Resume (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
job_file = st.file_uploader("📄 Upload Job Description (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

# Analyze Skills Button
if st.button("Analyze Skills"):
    if resume_file and job_file:
        resume_text = extract_text_from_file(resume_file)
        job_text = extract_text_from_file(job_file)

        if resume_text and job_text:
            resume_skills = extract_skills(resume_text)
            job_skills = extract_skills(job_text)
            missing_skills = find_missing_skills(resume_skills, job_skills)

            # Store results in session state
            st.session_state.skills_analyzed = True
            st.session_state.missing_skills = missing_skills

            st.subheader("🔍 Extracted Skills")
            st.write(f"**Resume Skills:** {', '.join(resume_skills)}")
            st.write(f"**Job Required Skills:** {', '.join(job_skills)}")

            st.subheader("⚠️ Missing Skills")
            if missing_skills:
                st.write(f"You are missing: {', '.join(missing_skills)}")
            else:
                st.success("You have all the required skills!")
    else:
        st.warning("Please upload both Resume and Job Description files before analyzing!")

# Get Recommended Courses Button
if st.session_state.skills_analyzed:
    if st.button("📚 Get Recommended Courses"):
        if st.session_state.missing_skills:
            all_courses = []
            for skill in st.session_state.missing_skills:
                courses = fetch_youtube_courses(skill)
                all_courses.extend(courses)
            
            if all_courses:
                df = pd.DataFrame(all_courses)
                st.table(df)  # Display courses as a table
            else:
                st.write("No courses found.")
        else:
            st.success("You have all the required skills! No additional courses needed.")
else:
    st.warning("Analyze skills first to get recommendations!")
