import streamlit as st
import pdfplumber
import docx
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

# Function to compare resume and job description
def compare_texts(resume_text, job_desc_text):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_desc_text])
    similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    return round(similarity_score * 100, 2)  # Convert to percentage

# Function to extract key skills
def extract_keywords(text):
    words = text.lower().split()
    keywords = [word for word in words if len(word) > 3]  # Filtering short words
    return list(set(keywords))[:10]  # Returning top 10 keywords

# Streamlit UI
st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

# Custom Styling
st.markdown("""
    <style>
        .stApp {background-color: #1e1e2f;}
        h1, h2, h3, p {color: white;}
        .stSidebar {background-color: #252535;}
    </style>
""", unsafe_allow_html=True)

st.title("📄 AI Resume Analyzer")
st.markdown("#### Upload your resume and job description to analyze compatibility.")

# Sidebar Upload Section
st.sidebar.header("📂 Upload Files")
resume_file = st.sidebar.file_uploader("Upload Resume (PDF/DOCX)", type=["pdf", "docx"])
job_desc_file = st.sidebar.file_uploader("Upload Job Description (TXT/DOCX)", type=["txt", "docx"])

if resume_file and job_desc_file:
    with st.spinner("🔍 Processing... Please wait"):
        # Extract text
        resume_text = extract_text_from_pdf(resume_file) if resume_file.name.endswith(".pdf") else extract_text_from_docx(resume_file)
        job_desc_text = extract_text_from_docx(job_desc_file) if job_desc_file.name.endswith(".docx") else job_desc_file.read().decode("utf-8")

        # Extract key skills
        resume_keywords = extract_keywords(resume_text)
        job_desc_keywords = extract_keywords(job_desc_text)

        # Compare and display results
        match_score = compare_texts(resume_text, job_desc_text)
        
        # Summary Section
        st.subheader("📌 Brief Summary")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📄 Resume Summary")
            st.write(f"**Top Skills Identified:** {', '.join(resume_keywords)}")
            st.write(f"**Total Words:** {len(resume_text.split())}")

        with col2:
            st.markdown("### 📜 Job Description Summary")
            st.write(f"**Required Skills:** {', '.join(job_desc_keywords)}")
            st.write(f"**Total Words:** {len(job_desc_text.split())}")

        # Visualization Section
        st.subheader("📊 Match Score Breakdown")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Pie Chart
        axes[0].pie([match_score, 100 - match_score], labels=["Match", "Mismatch"], autopct='%1.1f%%', colors=["skyblue", "lightgray"])
        axes[0].set_title("Resume vs. Job Description Match")

        # Bar Chart
        sns.barplot(x=["Match Score"], y=[match_score], ax=axes[1], palette="Blues")
        axes[1].set_ylim(0, 100)
        axes[1].set_ylabel("Percentage")
        axes[1].set_title("Resume Match Percentage")

        st.pyplot(fig)

        # Recommendations
        st.subheader("📌 Recommendations")
        missing_skills = list(set(job_desc_keywords) - set(resume_keywords))

        if match_score >= 80:
            st.success("✅ Your resume is a great match for this job! Keep it up.")
        elif 50 <= match_score < 80:
            st.warning(f"⚠️ Moderate match. Consider adding relevant skills: {', '.join(missing_skills)}.")
        else:
            st.error(f"❌ Your resume does not match well. Add missing skills: {', '.join(missing_skills)}.")

        # Skill Improvement Suggestions
        st.subheader("🎯 Suggested Courses for Missing Skills")
        for skill in missing_skills:
            course_url = f"https://www.udemy.com/courses/search/?q={skill.replace(' ', '%20')}"
            st.markdown(f"🔹 [{skill} Course]( {course_url} )")

        st.sidebar.info("🔹 Make sure your resume is optimized for best results.")
