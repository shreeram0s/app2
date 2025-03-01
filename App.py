import streamlit as st
import pandas as pd
import docx
import PyPDF2
import matplotlib.pyplot as plt
import re
from sentence_transformers import SentenceTransformer, util
from bs4 import BeautifulSoup
import requests

# Load SBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = "".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
    return text

# Function to extract text from DOCX
def extract_text_from_docx(uploaded_file):
    doc = docx.Document(uploaded_file)
    return "\n".join([para.text for para in doc.paragraphs])

# Function to extract text from TXT
def extract_text_from_txt(uploaded_file):
    return uploaded_file.read().decode("utf-8")

# Function to extract text based on file type
def extract_text(uploaded_file):
    if uploaded_file.name.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)
    elif uploaded_file.name.endswith(".docx"):
        return extract_text_from_docx(uploaded_file)
    elif uploaded_file.name.endswith(".txt"):
        return extract_text_from_txt(uploaded_file)
    else:
        return ""

# Function to extract skills using NLP techniques
def extract_skills(text):
    predefined_skills = {"Python", "SQL", "Power BI", "Machine Learning", "AWS", "Data Science"}  # Expand dynamically
    return {skill for skill in predefined_skills if re.search(rf"\b{skill}\b", text, re.IGNORECASE)}

# Function to compare resume and job description skills
def analyze_skills(resume_text, job_desc_text):
    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_desc_text)
    missing_skills = job_skills - resume_skills
    return list(missing_skills), list(resume_skills), list(job_skills)

# Function to fetch learning resources dynamically
def fetch_learning_resources(skill):
    search_url = f"https://www.google.com/search?q=best+online+course+for+{skill}"
    response = requests.get(search_url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(response.text, "html.parser")
    links = [a['href'] for a in soup.find_all("a", href=True) if "http" in a['href']]
    return links[:3]  # Return top 3 links

# Function to generate learning plan
def generate_learning_plan(missing_skills):
    schedule = []
    for day, skill in enumerate(missing_skills, start=1):
        resources = fetch_learning_resources(skill)
        for resource in resources:
            schedule.append((f"Day {day}", skill, f"[Click here]({resource})"))
    return pd.DataFrame(schedule, columns=["Day", "Skill", "Resource Link"])

# Streamlit UI
st.title("📄 AI Resume Analyzer - Skill Gap Learning Plan")

# Upload resume
resume_file = st.file_uploader("Upload your Resume (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
job_desc = st.text_area("Paste the Job Description")

if resume_file and job_desc:
    with st.spinner("Processing..."):
        resume_text = extract_text(resume_file)
        missing_skills, resume_skills, job_skills = analyze_skills(resume_text, job_desc)

        # Display extracted information
        st.subheader("🔎 Resume Content Preview")
        st.write(resume_text[:500] + "...")
        
        st.subheader("📄 Job Description Preview")
        st.write(job_desc[:500] + "...")
        
        # Skill matching visualization
        st.subheader("📊 Skill Matching Visualization")
        skill_labels = ["Matched Skills", "Missing Skills"]
        skill_counts = [len(resume_skills), len(missing_skills)]
        
        fig, ax = plt.subplots()
        ax.bar(skill_labels, skill_counts, color=["green", "red"])
        ax.set_ylabel("Count")
        ax.set_title("Resume vs. Job Description Skills")
        st.pyplot(fig)
        
        if missing_skills:
            st.warning(f"🚨 Missing Skills Identified: {', '.join(missing_skills)}")
            schedule_df = generate_learning_plan(missing_skills)
            
            st.subheader("📅 Personalized Learning Schedule")
            st.markdown("This schedule includes clickable learning resource links:")
            st.dataframe(schedule_df, width=800)
        else:
            st.success("✅ No missing skills detected! Your resume is well-matched.")
