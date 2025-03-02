import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Load the pre-trained NLP model for semantic comparison
model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to extract skills from a given text
def extract_skills(text):
    skills_db = ["Python", "SQL", "Java", "Power BI", "JavaScript", "Machine Learning", "Deep Learning", "Django", "Flask", "React", "AWS", "Azure", "Data Science"]
    text = text.lower()
    words = re.findall(r'\b\w+\b', text)
    extracted_skills = [skill for skill in skills_db if skill.lower() in words]
    return extracted_skills

# Function to compare resume and job description
def analyze_resume(resume_text, job_desc_text):
    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_desc_text)
    missing_skills = list(set(job_skills) - set(resume_skills))
    return resume_skills, job_skills, missing_skills

# Function to generate a structured learning plan (Placeholder)
def generate_learning_plan(missing_skills):
    schedule = []
    for i, skill in enumerate(missing_skills, start=1):
        schedule.append((f"Week {i}", skill, f"Search for {skill} courses online"))
    
    return pd.DataFrame(schedule, columns=["Week", "Skill", "Resource Recommendation"])

# Function to visualize skill comparison
def visualize_skills(resume_skills, job_skills, missing_skills):
    labels = ["Resume Skills", "Job Required Skills", "Missing Skills"]
    values = [len(resume_skills), len(job_skills), len(missing_skills)]
    
    fig, ax = plt.subplots()
    sns.barplot(x=labels, y=values, palette=["green", "blue", "red"], ax=ax)
    ax.set_title("Skill Comparison")
    st.pyplot(fig)

# Streamlit UI
st.title("📄 AI Resume Analyzer - Skill Gap Learning Plan")
st.subheader("📌 Upload your Resume and Paste the Job Description to analyze skill gaps!")

# Upload resume file
resume_file = st.file_uploader("Upload your Resume (Text File)", type=["txt"])
job_desc = st.text_area("Paste the Job Description")

if resume_file and job_desc:
    with st.spinner("Processing..."):
        resume_text = resume_file.read().decode("utf-8", errors="ignore")
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
            
            # Visualize skills
            visualize_skills(resume_skills, job_skills, missing_skills)
        else:
            st.success("✅ No missing skills detected! Your resume is well-matched.")
