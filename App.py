import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns
import re
import docx
import pdfplumber

# Load the pre-trained NLP model for semantic comparison
model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to extract text from different file formats
def extract_text_from_file(uploaded_file):
    file_extension = uploaded_file.name.split(".")[-1].lower()
    
    if file_extension == "txt":
        return uploaded_file.read().decode("utf-8", errors="ignore")
    elif file_extension == "docx":
        doc = docx.Document(uploaded_file)
        return "\n".join([para.text for para in doc.paragraphs])
    elif file_extension == "pdf":
        with pdfplumber.open(uploaded_file) as pdf:
            return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    else:
        return None

# Function to extract skills from text
def extract_skills(text):
    skills_db = ["Python", "SQL", "Java", "Power BI", "JavaScript", "Machine Learning", "Deep Learning", "Django", "Flask", "React", "AWS", "Azure", "Data Science"]
    text = text.lower()
    words = re.findall(r'\b\w+\b', text)
    extracted_skills = [skill for skill in skills_db if skill.lower() in words]
    return extracted_skills

# Function to generate a summary
def summarize_text(text, num_sentences=3):
    sentences = text.split(". ")
    return ". ".join(sentences[:num_sentences]) + "..." if len(sentences) > num_sentences else text

# Function to compare resume and job description
def analyze_resume(resume_text, job_desc_text):
    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_desc_text)
    missing_skills = list(set(job_skills) - set(resume_skills))
    return resume_skills, job_skills, missing_skills

# Function to fetch learning resources dynamically
def fetch_learning_resources(skill):
    search_url = f"https://www.google.com/search?q={skill}+online+course+site:udemy.com+OR+site:coursera.org+OR+site:edx.org"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    
    links = []
    for link in soup.find_all("a", href=True):
        url = link["href"]
        if "http" in url and ("udemy.com" in url or "coursera.org" in url or "edx.org" in url):
            links.append(url)
        if len(links) >= 3:  # Limit to top 3 relevant links
            break
    
    return links[:3]

# Function to generate a structured learning plan
def generate_learning_plan(missing_skills):
    schedule = []
    for i, skill in enumerate(missing_skills, start=1):
        resources = fetch_learning_resources(skill)
        clickable_links = [f"[Course {j+1}]({link})" for j, link in enumerate(resources)]
        schedule.append((f"Week {i}", skill, ", ".join(clickable_links) if clickable_links else "No resources found"))
    
    return pd.DataFrame(schedule, columns=["Week", "Skill", "Recommended Courses"])

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
resume_file = st.file_uploader("Upload your Resume (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
job_desc_file = st.file_uploader("Upload Job Description (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

if resume_file and job_desc_file:
    with st.spinner("Processing..."):
        resume_text = extract_text_from_file(resume_file)
        job_desc_text = extract_text_from_file(job_desc_file)
        
        if resume_text and job_desc_text:
            # Display summaries
            st.subheader("📄 Resume Summary")
            st.write(summarize_text(resume_text))
            
            st.subheader("📝 Job Description Summary")
            st.write(summarize_text(job_desc_text))
            
            # Analyze skills
            resume_skills, job_skills, missing_skills = analyze_resume(resume_text, job_desc_text)
            
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
        else:
            st.error("⚠️ Unable to extract text from one or both files. Please check the formats.")
