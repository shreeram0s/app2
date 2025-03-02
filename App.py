import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer

# Load NLP model for semantic comparison
model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_skills(text):
    """Extract skills dynamically from text using NLP matching."""
    skills_db = ["Python", "SQL", "Java", "Power BI", "JavaScript", "Machine Learning", "Deep Learning", "Django", "Flask", "React", "AWS", "Azure", "Data Science"]
    extracted_skills = [skill for skill in skills_db if skill.lower() in text.lower()]
    return extracted_skills

def analyze_resume(resume_text, job_desc_text):
    """Compare resume and job description, identifying missing skills."""
    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_desc_text)
    missing_skills = list(set(job_skills) - set(resume_skills))
    matching_score = len(set(resume_skills) & set(job_skills)) / len(job_skills) * 100 if job_skills else 0
    return resume_skills, job_skills, missing_skills, round(matching_score, 2)

def fetch_learning_resources(skill):
    """Fetch accessible online course links dynamically for each skill."""
    search_url = f"https://www.google.com/search?q={skill}+best+online+courses"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    links = [a['href'] for a in soup.find_all("a", href=True) if "http" in a['href'] and "google" not in a['href']][:5]
    
    return links

def generate_learning_plan(missing_skills):
    """Generate a structured study schedule with real online courses."""
    schedule = []
    for skill in missing_skills:
        resources = fetch_learning_resources(skill)
        for i, resource in enumerate(resources, start=1):
            schedule.append((f"Day {i}", skill, resource))
    return pd.DataFrame(schedule, columns=["Day", "Skill", "Course Link"])

# Streamlit UI
st.title("📄 AI Resume Analyzer - Skill Gap Learning Plan")
st.subheader("📌 Upload your Resume and Paste the Job Description")

# Upload resume
resume_file = st.file_uploader("Upload Resume (Text File)", type=["txt"])
job_desc = st.text_area("Paste Job Description")

if resume_file and job_desc:
    resume_text = resume_file.read().decode("utf-8")
    st.subheader("🔍 Brief Description of Uploaded Resume and Job Description")
    st.write(f"**Resume Preview:** {resume_text[:300]}...")
    st.write(f"**Job Description Preview:** {job_desc[:300]}...")
    
    with st.spinner("Processing..."):
        resume_skills, job_skills, missing_skills, match_score = analyze_resume(resume_text, job_desc)
        
        st.subheader("📊 Skill Analysis & Matching Score")
        st.write(f"✅ **Skills in Resume:** {', '.join(resume_skills)}")
        st.write(f"🎯 **Skills Required:** {', '.join(job_skills)}")
        st.metric("Resume Matching Score", f"{match_score}%")
        
        if missing_skills:
            st.warning(f"🚀 **You are missing these skills:** {', '.join(missing_skills)}")
            
            # Generate structured learning plan
            schedule_df = generate_learning_plan(missing_skills)
            st.subheader("📅 Personalized Study Plan")
            st.dataframe(schedule_df)
            
            # Visualization
            st.subheader("📈 Visualization of Skills Matching")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(x=["Matching", "Missing"], y=[len(resume_skills), len(missing_skills)], palette=["green", "red"], ax=ax)
            ax.set_ylabel("Skill Count")
            ax.set_title("Resume vs Job Skill Comparison")
            st.pyplot(fig)
        else:
            st.success("✅ No missing skills detected! Your resume is well-matched.")
