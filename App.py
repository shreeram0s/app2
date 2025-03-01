import streamlit as st
import pdfplumber
import docx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Predefined set of industry-relevant skills
COMMON_SKILLS = {
    "python", "java", "c++", "machine learning", "deep learning", "data analysis", 
    "excel", "sql", "power bi", "tableau", "cloud computing", "aws", "gcp", "azure", 
    "nlp", "tensorflow", "keras", "pandas", "numpy", "scikit-learn", "data visualization"
}

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    with pdfplumber.open(uploaded_file) as pdf:
        text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    return text

# Function to extract text from DOCX
def extract_text_from_docx(uploaded_file):
    doc = docx.Document(uploaded_file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

# Function to clean text and extract only skills
def extract_skills(text):
    text = text.lower()  # Convert text to lowercase
    words = re.findall(r'\b[a-zA-Z-]+\b', text)  # Extract only words
    skills = set(words).intersection(COMMON_SKILLS)  # Filter relevant skills
    return list(skills)

# Function to compare resume and job description
def compare_texts(resume_text, job_desc_text):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_desc_text])
    similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    return round(similarity_score * 100, 2)  # Convert to percentage

# Function to suggest courses based on missing skills
def suggest_courses(missing_skills):
    course_links = {}
    for skill in missing_skills:
        query = f"{skill} online course"
        response = requests.get(f"https://www.google.com/search?q={query}")
        course_links[skill] = f"https://www.google.com/search?q={query}"  # Google search link
    return course_links

# Streamlit UI
st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

st.title("📄 AI Resume Analyzer")
st.markdown("#### Upload your resume and job description to analyze compatibility.")

# Sidebar Upload Section
st.sidebar.header("📂 Upload Files")
resume_file = st.sidebar.file_uploader("Upload Resume (PDF/DOCX)", type=["pdf", "docx"])
job_desc_file = st.sidebar.file_uploader("Upload Job Description (TXT/DOCX)", type=["txt", "docx"])

if resume_file and job_desc_file:
    with st.spinner("🔍 Processing... Please wait"):
        # Extract text from files
        resume_text = extract_text_from_pdf(resume_file) if resume_file.name.endswith(".pdf") else extract_text_from_docx(resume_file)
        job_desc_text = extract_text_from_docx(job_desc_file) if job_desc_file.name.endswith(".docx") else job_desc_file.read().decode("utf-8")

        # Extract skills
        resume_skills = extract_skills(resume_text)
        job_desc_skills = extract_skills(job_desc_text)

        # Find missing skills
        missing_skills = list(set(job_desc_skills) - set(resume_skills))
        
        # Compare and display results
        match_score = compare_texts(resume_text, job_desc_text)

        st.subheader("📋 Resume & Job Description Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🔹 Resume Overview**")
            st.write(f"**Top Skills Identified:** {', '.join(resume_skills) if resume_skills else 'No skills detected'}")
            st.write(f"**Total Words:** {len(resume_text.split())}")
        
        with col2:
            st.markdown("**🔹 Job Description Overview**")
            st.write(f"**Required Skills:** {', '.join(job_desc_skills) if job_desc_skills else 'No skills detected'}")
            st.write(f"**Total Words:** {len(job_desc_text.split())}")

        # Visualization of match score
        st.subheader("📊 Match Score Breakdown")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(["Match Score"], [match_score], color="skyblue")
        ax.set_ylim(0, 100)
        ax.set_ylabel("Percentage")
        ax.set_title("Resume vs. Job Description Match")
        st.pyplot(fig)

        # Pie Chart Visualization
        st.subheader("📊 Skill Match Breakdown")
        labels = ["Matched Skills", "Missing Skills"]
        sizes = [len(set(resume_skills).intersection(job_desc_skills)), len(missing_skills)]
        colors = ["green", "red"]
        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140)
        st.pyplot(fig)

        # Recommendations
        st.subheader("📌 Recommendations")
        if match_score >= 80:
            st.success("✅ Your resume is a great match for this job! Keep it up.")
        elif 50 <= match_score < 80:
            st.warning("⚠️ Moderate match. Consider adding relevant skills and keywords.")
        else:
            st.error("❌ Your resume does not match well. Update it with relevant skills.")

        # Course Suggestions for Missing Skills
        if missing_skills:
            st.subheader("🎓 Suggested Courses to Improve Your Resume")
            courses = suggest_courses(missing_skills)
            for skill, link in courses.items():
                st.markdown(f"✅ **{skill.capitalize()}**: [Find courses here]({link})")

st.sidebar.info("🔹 Make sure your resume is optimized for best results.")
