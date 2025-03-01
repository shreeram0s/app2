import streamlit as st
import pandas as pd
import re
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util

# Load SBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_skills(text):
    """Extracts skills from text using regex matching."""
    skills = {"SQL", "Power BI", "Python", "AWS", "Machine Learning", "Data Science"}  # Expand this list
    found_skills = {skill for skill in skills if re.search(rf"\b{skill}\b", text, re.IGNORECASE)}
    return found_skills

def find_missing_skills(resume_text, job_desc_text):
    """Identifies missing skills by comparing resume and job description."""
    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_desc_text)
    missing_skills = job_skills - resume_skills
    return list(missing_skills)

def fetch_learning_resources(skill):
    """Dynamically fetches online courses for a given skill."""
    search_url = f"https://www.google.com/search?q={skill}+online+course"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    links = []
    for link in soup.find_all("a", href=True):
        url = link["href"]
        if "http" in url and "google" not in url:
            links.append(url)
        if len(links) >= 3:
            break
    return links[:3] if links else ["No resources found"]

def generate_learning_plan(missing_skills):
    """Generates a structured learning plan with clickable links."""
    schedule = []
    day_counter = 1
    for skill in missing_skills:
        resources = fetch_learning_resources(skill)
        for resource in resources:
            schedule.append((f"Day {day_counter}", skill, f"[Click Here]({resource})"))
            day_counter += 1
    return pd.DataFrame(schedule, columns=["Day", "Skill", "Resource Link"])

# Streamlit UI
st.title("📄 AI Resume Analyzer - Skill Gap Learning Plan")
st.subheader("📌 Upload your resume and job description to identify skill gaps!")

resume_text = st.text_area("Paste Your Resume Text")
job_desc_text = st.text_area("Paste the Job Description")

if resume_text and job_desc_text:
    missing_skills = find_missing_skills(resume_text, job_desc_text)
    if missing_skills:
        st.success(f"🔍 Missing Skills Identified: {', '.join(missing_skills)}")
        schedule_df = generate_learning_plan(missing_skills)
        st.subheader("📅 Personalized Learning Schedule")
        st.dataframe(schedule_df, use_container_width=True)
    else:
        st.success("✅ No missing skills detected! Your resume is well-matched.")
