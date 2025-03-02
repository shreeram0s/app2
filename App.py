import streamlit as st
import requests
import docx2txt
import pdfplumber
import re
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import numpy as np

# Load sentence transformer model for skill extraction
model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_text_from_file(uploaded_file):
    if uploaded_file.type == "application/pdf":
        with pdfplumber.open(uploaded_file) as pdf:
            return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return docx2txt.process(uploaded_file)
    elif uploaded_file.type == "text/plain":
        return uploaded_file.read().decode("utf-8")
    return ""

def extract_skills(text):
    words = text.lower().split()
    embeddings = model.encode(words)
    unique_skills = set(words)  # Simplified skill extraction
    return list(unique_skills)

def find_missing_skills(resume_text, job_desc_text):
    resume_skills = set(extract_skills(resume_text))
    job_skills = set(extract_skills(job_desc_text))
    missing_skills = job_skills - resume_skills
    return list(missing_skills)

def fetch_course_links(skill):
    search_url = f"https://www.google.com/search?q={skill}+online+course"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(search_url, headers=headers)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        links = []
        for link in soup.find_all("a", href=True):
            href = link.get("href")
            if "url?q=" in href:
                match = re.search(r'url\?q=(.*?)&', href)
                if match:
                    clean_url = match.group(1)
                    if "google.com" not in clean_url:
                        links.append(clean_url)
        return links[:3]  # Return top 3 course links
    return []

def plot_skill_gap(missing_skills):
    if not missing_skills:
        st.write("✅ No skill gaps detected!")
        return
    skill_counts = np.random.randint(1, 10, size=len(missing_skills))
    plt.figure(figsize=(8, 5))
    plt.barh(missing_skills, skill_counts, color='skyblue')
    plt.xlabel("Importance Level")
    plt.ylabel("Missing Skills")
    plt.title("Skill Gap Analysis")
    st.pyplot(plt)

st.title("AI Resume Skill Analyzer & Course Recommender")

resume_file = st.file_uploader("Upload Resume (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
job_file = st.file_uploader("Upload Job Description (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

if resume_file and job_file:
    resume_text = extract_text_from_file(resume_file)
    job_desc_text = extract_text_from_file(job_file)
    missing_skills = find_missing_skills(resume_text, job_desc_text)
    
    st.subheader("Skill Gap Analysis")
    plot_skill_gap(missing_skills)
    
    st.subheader("Recommended Courses")
    for skill in missing_skills:
        courses = fetch_course_links(skill)
        if courses:
            st.write(f"### {skill}")
            for course in courses:
                st.markdown(f"- [Course Link]({course})")
        else:
            st.write(f"No online courses found for {skill}")
