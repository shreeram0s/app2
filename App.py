import streamlit as st
import pandas as pd
import io
import re
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util

# Load SBERT model for semantic similarity
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to extract skills using regex
def extract_skills(text):
    skills = {"SQL", "Power BI", "Python", "Java", "JavaScript", "HTML", "CSS"}
    found_skills = {skill for skill in skills if re.search(rf"\\b{skill}\\b", text, re.IGNORECASE)}
    return found_skills

# Function to identify missing skills
def find_missing_skills(resume_text, job_desc_text):
    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_desc_text)
    missing_skills = job_skills - resume_skills
    return list(missing_skills)

# Function to fetch learning resources dynamically using web scraping
def fetch_learning_resources(skill):
    search_url = f"https://www.google.com/search?q=best+courses+for+{skill}+site:udemy.com+OR+site:coursera.org+OR+site:datacamp.com"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    
    links = []
    for g in soup.find_all('a', href=True):
        url = g['href']
        if "udemy.com" in url or "coursera.org" in url or "datacamp.com" in url:
            links.append(url)
        if len(links) >= 3:
            break
    
    return links if links else ["No suitable resources found"]

# Function to generate a structured learning plan
def generate_learning_plan(missing_skills):
    schedule = []
    for skill in missing_skills:
        resources = fetch_learning_resources(skill)
        for i, resource in enumerate(resources):
            schedule.append((f"Day {i+1}", skill, f'<a href="{resource}" target="_blank">{resource}</a>'))
    return pd.DataFrame(schedule, columns=["Day", "Skill", "Resource Link"])

# Streamlit UI
st.title("📄 AI Resume Analyzer - Skill Gap Learning Plan")
st.subheader("📌 Upload your resume and paste job description to identify skill gaps!")

# Upload resume
resume_file = st.file_uploader("Upload your Resume (TXT, PDF)", type=["txt"])
job_desc = st.text_area("Paste the Job Description")

if resume_file and job_desc:
    with st.spinner("Processing..."):
        resume_text = resume_file.getvalue().decode("utf-8")
        missing_skills = find_missing_skills(resume_text, job_desc)

        if missing_skills:
            st.success(f"🔍 Missing Skills Identified: {', '.join(missing_skills)}")
            schedule_df = generate_learning_plan(missing_skills)
            st.subheader("📅 Personalized Learning Schedule")
            st.markdown(schedule_df.to_html(escape=False, index=False), unsafe_allow_html=True)
        else:
            st.success("✅ No missing skills detected! Your resume is well-matched.")
