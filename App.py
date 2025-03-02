import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import re
import docx
import pdfplumber
import matplotlib.pyplot as plt

# Load NLP model
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
    return None

# Function to extract skills from text
def extract_skills(text):
    skills_db = ["Python", "SQL", "Java", "Power BI", "JavaScript", "Machine Learning", "Deep Learning", "Django", "Flask", "React", "AWS", "Azure", "Data Science"]
    text = text.lower()
    words = re.findall(r'\b\w+\b', text)
    return [skill for skill in skills_db if skill.lower() in words]

# Function to compare resume and job description
def analyze_resume(resume_text, job_desc_text):
    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_desc_text)
    missing_skills = list(set(job_skills) - set(resume_skills))
    return resume_skills, job_skills, missing_skills

# Function to fetch learning resources dynamically from Google Search using BeautifulSoup
def fetch_learning_resources(skill):
    headers = {"User-Agent": "Mozilla/5.0"}
    search_url = f"https://www.google.com/search?q={skill}+online+course+site:udemy.com+OR+site:coursera.org+OR+site:edx.org"

    try:
        response = requests.get(search_url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        
        links = []
        for g in soup.find_all("a", href=True):
            href = g["href"]
            if "url?q=" in href and not "webcache" in href:
                link = href.split("url?q=")[1].split("&")[0]
                if "udemy.com" in link or "coursera.org" in link or "edx.org" in link:
                    links.append(link)
                    
        return links[:3]  # Return top 3 results
    except Exception as e:
        return []

# Function to generate a structured learning plan
def generate_learning_plan(missing_skills):
    schedule = []
    for i, skill in enumerate(missing_skills, start=1):
        resources = fetch_learning_resources(skill)
        clickable_links = [f"[Course {j+1}]({link})" for j, link in enumerate(resources)]
        schedule.append((f"Week {i}", skill, ", ".join(clickable_links) if clickable_links else "No resources found"))
    
    return pd.DataFrame(schedule, columns=["Week", "Skill", "Recommended Courses"])

# Function to visualize skill gaps using a bar chart
def plot_skill_gap(resume_skills, job_skills):
    all_skills = set(resume_skills + job_skills)
    resume_counts = [1 if skill in resume_skills else 0 for skill in all_skills]
    job_counts = [1 if skill in job_skills else 0 for skill in all_skills]

    df = pd.DataFrame({"Skill": list(all_skills), "Resume": resume_counts, "Job Description": job_counts})
    df.set_index("Skill", inplace=True)
    df.plot(kind="bar", figsize=(8, 4), color=["skyblue", "salmon"])
    
    plt.title("Skill Comparison")
    plt.ylabel("Presence (1 = Found, 0 = Not Found)")
    plt.xticks(rotation=45)
    st.pyplot(plt)

# Streamlit UI
st.title("📄 AI Resume Analyzer - Skill Gap Learning Plan")

# Upload files
resume_file = st.file_uploader("Upload your Resume", type=["pdf", "docx", "txt"])
job_desc_file = st.file_uploader("Upload Job Description", type=["pdf", "docx", "txt"])

if resume_file and job_desc_file:
    with st.spinner("Processing..."):
        resume_text = extract_text_from_file(resume_file)
        job_desc_text = extract_text_from_file(job_desc_file)
        
        if resume_text and job_desc_text:
            resume_skills, job_skills, missing_skills = analyze_resume(resume_text, job_desc_text)

            # Display summary of resume and job description
            st.subheader("📄 Resume Summary")
            st.write(resume_text[:500] + "...")  # Display first 500 characters
            
            st.subheader("📜 Job Description Summary")
            st.write(job_desc_text[:500] + "...")  # Display first 500 characters
            
            # Display skill comparison
            st.subheader("📌 Skill Comparison")
            plot_skill_gap(resume_skills, job_skills)

            st.subheader("🚀 Missing Skills")
            if missing_skills:
                st.warning(f"You need to learn: {', '.join(missing_skills)}")

                # Generate learning plan
                schedule_df = generate_learning_plan(missing_skills)
                st.subheader("📅 Personalized Learning Schedule")
                st.dataframe(schedule_df)
            else:
                st.success("✅ No missing skills detected!")
        else:
            st.error("⚠️ Unable to extract text from files.")
