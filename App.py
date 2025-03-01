import streamlit as st
import pandas as pd
import io
import PyPDF2
import docx
import re
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
from bs4 import BeautifulSoup
import requests

# Load the SBERT model
model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_text_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = "".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
    return text

def extract_text_from_docx(uploaded_file):
    doc = docx.Document(uploaded_file)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text(uploaded_file):
    if uploaded_file.type == "application/pdf":
        return extract_text_from_pdf(uploaded_file)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return extract_text_from_docx(uploaded_file)
    elif uploaded_file.type.startswith("text"):
        return str(uploaded_file.read(), "utf-8")
    return ""

def extract_skills(text):
    skills = {"SQL", "Power BI", "Python", "AWS", "Machine Learning", "Data Science"}
    return {skill for skill in skills if re.search(rf"\\b{skill}\\b", text, re.IGNORECASE)}

def find_missing_skills(resume_text, job_desc_text):
    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_desc_text)
    missing_skills = job_skills - resume_skills
    return list(missing_skills), list(resume_skills), list(job_skills)

def fetch_learning_resources(skill):
    search_url = f"https://www.google.com/search?q=best+{skill}+courses"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    links = [a['href'] for a in soup.find_all('a', href=True) if "http" in a['href'] and "google.com" not in a['href']]
    return links[:5] if links else ["No resources found"]

def generate_learning_plan(missing_skills):
    schedule = []
    day_counter = 1
    for skill in missing_skills:
        resources = fetch_learning_resources(skill)
        for resource in resources:
            schedule.append((f"Day {day_counter}", skill, f"[Click here]({resource})"))
        day_counter += 1
    df = pd.DataFrame(schedule, columns=["Day", "Skill", "Resource Link"])
    return df.drop_duplicates()

st.title("📄 AI Resume Analyzer - Skill Gap Learning Plan")
st.subheader("📌 Upload your resume and job description to identify skill gaps!")

resume_file = st.file_uploader("Upload your Resume (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
job_desc_file = st.file_uploader("Upload the Job Description (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

if resume_file and job_desc_file:
    with st.spinner("Processing..."):
        resume_text = extract_text(resume_file)
        job_desc_text = extract_text(job_desc_file)
        missing_skills, resume_skills, job_skills = find_missing_skills(resume_text, job_desc_text)
        
        st.subheader("🔎 Resume Summary")
        st.write(resume_text[:500] + "...")
        
        st.subheader("📄 Job Description Overview")
        st.write(job_desc_text[:500] + "...")
        
        st.subheader("📊 Skill Matching Visualization")
        skill_labels = ["Matched Skills", "Missing Skills"]
        skill_counts = [len(resume_skills), len(missing_skills)]
        fig, ax = plt.subplots()
        ax.bar(skill_labels, skill_counts, color=["green", "red"])
        ax.set_ylabel("Count")
        ax.set_title("Resume vs. Job Description Skills")
        st.pyplot(fig)
        
        if missing_skills:
            st.success(f"🔍 Missing Skills Identified: {', '.join(missing_skills)}")
            st.subheader("📌 Recommendations")
            st.write("To improve your resume, consider learning these missing skills through the suggested courses below.")
            schedule_df = generate_learning_plan(missing_skills)
            st.subheader("📅 Personalized Learning Schedule")
            st.write("This schedule includes clickable learning resource links:")
            st.dataframe(schedule_df, hide_index=True, use_container_width=True)
        else:
            st.success("✅ No missing skills detected! Your resume is well-matched.")
