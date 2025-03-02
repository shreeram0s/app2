import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util

# Load the pre-trained NLP model for semantic comparison
model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to extract skills from a given text using NLP
def extract_skills(text):
    skills_db = ["Python", "SQL", "Java", "Power BI", "JavaScript", "Machine Learning", "Deep Learning", "Django", "Flask", "React", "AWS", "Azure", "Data Science"]
    extracted_skills = [skill for skill in skills_db if skill.lower() in text.lower()]
    return extracted_skills

# Function to compare resume and job description
def analyze_resume(resume_text, job_desc_text):
    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_desc_text)
    missing_skills = list(set(job_skills) - set(resume_skills))
    
    return resume_skills, job_skills, missing_skills

# Function to fetch learning resources dynamically from online sources
def fetch_learning_resources(skill):
    search_url = f"https://www.google.com/search?q={skill}+online+courses"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    
    links = []
    for link in soup.find_all("a", href=True):
        url = link["href"]
        if "http" in url and "google" not in url:  # Avoid Google internal links
            links.append(url)
        if len(links) >= 5:
            break
    
    return links[:5]  # Return top 5 links

# Function to generate a structured learning plan
def generate_learning_plan(missing_skills):
    schedule = []
    for skill in missing_skills:
        resources = fetch_learning_resources(skill)
        for day, resource in enumerate(resources, start=1):
            schedule.append((f"Day {day}", skill, resource))
    
    return pd.DataFrame(schedule, columns=["Day", "Skill", "Resource Link"])

# Streamlit UI
st.title("📄 AI Resume Analyzer - Skill Gap Learning Plan")
st.subheader("📌 Upload your Resume and Paste the Job Description to analyze skill gaps!")

# Upload resume file
resume_file = st.file_uploader("Upload your Resume (Text File)", type=["txt"])
job_desc = st.text_area("Paste the Job Description")

if resume_file and job_desc:
    with st.spinner("Processing..."):
        resume_text = resume_file.read().decode("utf-8")
        resume_skills, job_skills, missing_skills = analyze_resume(resume_text, job_desc)

        st.subheader("📌 Identified Skills")
        st.write(f"✅ **Skills in Resume:** {', '.join(resume_skills)}")
        st.write(f"🎯 **Skills Required in Job:** {', '.join(job_skills)}")

        if missing_skills:
            st.warning(f"🚀 **You are missing these skills:** {', '.join(missing_skills)}")
            
            # Generate structured learning plan
            schedule_df = generate_learning_plan(missing_skills)
            st.subheader("📅 Personalized Learning Schedule")
            st.dataframe(schedule_df)

        else:
            st.success("✅ No missing skills detected! Your resume is well-matched.")
