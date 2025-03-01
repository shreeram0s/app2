import streamlit as st
import pdfplumber
import docx
import time
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from bs4 import BeautifulSoup

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

# Extract key skills from job description
def extract_skills(text):
    skills = [
        "Python", "Machine Learning", "Deep Learning", "Data Science", "SQL", "Excel", 
        "Tableau", "Power BI", "NLP", "Computer Vision", "Flask", "Streamlit", "TensorFlow", 
        "Pandas", "NumPy", "Matplotlib", "Statistics", "AWS", "Docker", "Big Data", "Kubernetes"
    ]
    extracted_skills = [skill for skill in skills if skill.lower() in text.lower()]
    return extracted_skills

# Function to find missing skills
def find_missing_skills(resume_skills, job_skills):
    return list(set(job_skills) - set(resume_skills))

# Function to get course links
def get_course_links(skill):
    search_url = f"https://www.udemy.com/courses/search/?q={skill}"
    return search_url

# Streamlit UI
st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

st.title("📄 AI Resume Analyzer")
st.markdown("### 🔍 Upload your resume and job description to analyze compatibility.")

# Sidebar Upload Section
st.sidebar.header("📂 Upload Files")
resume_file = st.sidebar.file_uploader("📄 Upload Resume (PDF/DOCX)", type=["pdf", "docx"])
job_desc_file = st.sidebar.file_uploader("📑 Upload Job Description (TXT/DOCX)", type=["txt", "docx"])

st.sidebar.info("💡 **Tip:** Ensure your resume highlights key skills required for the job!")

if resume_file and job_desc_file:
    with st.spinner("🔄 Processing... Please wait"):
        time.sleep(1.5)

        # Extract text
        resume_text = extract_text_from_pdf(resume_file) if resume_file.name.endswith(".pdf") else extract_text_from_docx(resume_file)
        job_desc_text = extract_text_from_docx(job_desc_file) if job_desc_file.name.endswith(".docx") else job_desc_file.read().decode("utf-8")

        # Compute similarity
        match_score = compare_texts(resume_text, job_desc_text)
        
        st.subheader("📊 Resume Analysis")
        st.write(f"**Match Score:** `{match_score}%`")

        # Progress Bar
        progress_bar = st.progress(0)
        for percent in range(0, int(match_score) + 1, 5):
            time.sleep(0.05)
            progress_bar.progress(percent)

        # Visualization
        st.subheader("📌 Match Score Breakdown")
        fig, ax = plt.subplots(figsize=(6, 3))
        colors = ['#ff595e' if match_score < 50 else '#ffca3a' if match_score < 80 else '#8ac926']
        ax.bar(["Match Score"], [match_score], color=colors)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Percentage")
        ax.set_title("Resume vs. Job Description Match", fontsize=12, fontweight='bold')
        plt.xticks(fontsize=10)
        st.pyplot(fig)

        # Recommendations
        st.subheader("📌 Recommendations")
        if match_score >= 80:
            st.success("✅ **Your resume is a great match for this job!** Keep it up.")
        elif 50 <= match_score < 80:
            st.warning("⚠️ **Moderate match.** Consider adding relevant skills and keywords.")
        else:
            st.error("❌ **Your resume does not match well.** Update it with relevant skills.")

        # Skill Analysis
        st.subheader("📌 Skill Gap Analysis")
        resume_skills = extract_skills(resume_text)
        job_skills = extract_skills(job_desc_text)
        missing_skills = find_missing_skills(resume_skills, job_skills)

        if missing_skills:
            st.warning("🔻 Your resume is missing these important skills:")
            for skill in missing_skills:
                course_link = get_course_links(skill)
                st.markdown(f"🔹 **{skill}** - [Find Courses]({course_link})")
        else:
            st.success("✅ Your resume covers all the required skills!")

        # Optimization Tips
        with st.expander("💡 Optimization Tips"):
            st.write("✅ Include key skills mentioned in the job description.")
            st.write("✅ Use action words and quantified achievements.")
            st.write("✅ Align your experience with job requirements.")
