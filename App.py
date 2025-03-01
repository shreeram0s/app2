import streamlit as st
import pdfplumber
import docx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
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

# Streamlit UI
st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

# Custom Styling
st.markdown("""
    <style>
        .stApp { background: linear-gradient(135deg, #1f1c2c, #928DAB); color: white; }
        .stSidebar { background-color: #252535; padding: 10px; }
        h1, h2, h3, h4, h5, h6 { color: #ffffff; }
        .uploadedFile { border: 2px dashed #ffffff; padding: 10px; text-align: center; border-radius: 10px; }
        .css-1d391kg { background-color: transparent; }
        .stButton button { background-color: #4CAF50; color: white; padding: 10px 15px; border-radius: 5px; }
        .stButton button:hover { background-color: #45a049; }
    </style>
""", unsafe_allow_html=True)

st.title("📄 AI Resume Analyzer")
st.markdown("### 🔍 Upload your resume and job description to analyze compatibility.")

# Sidebar Upload Section
st.sidebar.header("📂 Upload Files")
resume_file = st.sidebar.file_uploader("📄 Upload Resume (PDF/DOCX)", type=["pdf", "docx"])
job_desc_file = st.sidebar.file_uploader("📑 Upload Job Description (TXT/DOCX)", type=["txt", "docx"])

st.sidebar.info("💡 **Tip:** Make sure your resume is well-optimized with relevant skills!")

# Process uploaded files
if resume_file and job_desc_file:
    with st.spinner("🔄 Processing... Please wait"):
        time.sleep(1.5)  # Simulate processing time

        # Extract text
        resume_text = extract_text_from_pdf(resume_file) if resume_file.name.endswith(".pdf") else extract_text_from_docx(resume_file)
        job_desc_text = extract_text_from_docx(job_desc_file) if job_desc_file.name.endswith(".docx") else job_desc_file.read().decode("utf-8")

        # Compute similarity
        match_score = compare_texts(resume_text, job_desc_text)
        
        st.subheader("📊 Resume Analysis")
        st.write(f"**Match Score:** `{match_score}%`")

        # Animated Progress Bar
        progress_bar = st.progress(0)
        for percent in range(0, int(match_score) + 1, 5):
            time.sleep(0.05)
            progress_bar.progress(percent)

        # Visualization with modern styling
        st.subheader("📌 Match Score Breakdown")
        fig, ax = plt.subplots(figsize=(6, 3))
        colors = ['#ff595e' if match_score < 50 else '#ffca3a' if match_score < 80 else '#8ac926']
        ax.bar(["Match Score"], [match_score], color=colors)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Percentage")
        ax.set_title("Resume vs. Job Description Match", fontsize=12, fontweight='bold')
        plt.xticks(fontsize=10)
        st.pyplot(fig)

        # Recommendations with better visuals
        st.subheader("📌 Recommendations")
        if match_score >= 80:
            st.success("✅ **Your resume is a great match for this job!** Keep it up.")
        elif 50 <= match_score < 80:
            st.warning("⚠️ **Moderate match.** Consider adding relevant skills and keywords.")
        else:
            st.error("❌ **Your resume does not match well.** Update it with relevant skills.")

        # Optimization Tips
        with st.expander("💡 Optimization Tips"):
            st.write("✅ Include key skills mentioned in the job description.")
            st.write("✅ Use action words and quantified achievements.")
            st.write("✅ Align your experience with job requirements.")

