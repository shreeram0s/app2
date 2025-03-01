import streamlit as st
import pdfplumber
import docx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string

# Predefined list of common skills (expand this as needed)
SKILLS_DB = [
    "python", "java", "machine learning", "deep learning", "data science", "sql", "excel",
    "communication", "project management", "cloud computing", "tensorflow", "pandas",
    "numpy", "nlp", "power bi", "tableau", "flask", "django", "big data", "data analytics"
]

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

# Function to process text: tokenization and removing stopwords
def process_text(text):
    stopwords = set([
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
        "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
        "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
        "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having",
        "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while",
        "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after",
        "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then",
        "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other",
        "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just",
        "don", "should", "now", "d", "ll", "m", "o", "re", "ve", "y", "ain", "aren", "couldn", "didn", "doesn", "hadn", "hasn",
        "haven", "isn", "ma", "mightn", "mustn", "needn", "shan", "shouldn", "wasn", "weren", "won", "wouldn"
    ])
    
    # Convert text to lowercase and remove punctuation
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    
    # Tokenize and remove stopwords
    words = text.split()
    words = [word for word in words if word not in stopwords]
    
    return words

# Function to extract skills from text
def extract_skills(text):
    words = process_text(text)
    extracted_skills = list(set(word for word in words if word in SKILLS_DB))
    return extracted_skills

# Function to compare resume and job description
def compare_texts(resume_text, job_desc_text):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_desc_text])
    similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    return round(similarity_score * 100, 2)  # Convert to percentage

# Function to suggest courses for missing skills
def suggest_courses(missing_skills):
    course_links = {
        "python": "https://www.coursera.org/courses?query=python",
        "machine learning": "https://www.coursera.org/courses?query=machine%20learning",
        "data science": "https://www.udacity.com/course/data-scientist-nanodegree--nd025",
        "sql": "https://www.datacamp.com/courses/intro-to-sql-for-data-science",
        "excel": "https://www.udemy.com/course/microsoft-excel/",
        "deep learning": "https://www.deeplearning.ai/",
        "nlp": "https://www.coursera.org/specializations/natural-language-processing",
        "power bi": "https://www.udemy.com/course/power-bi-course/",
        "tableau": "https://www.tableau.com/learn/training",
        "cloud computing": "https://aws.amazon.com/training/",
    }
    suggested_courses = {skill: course_links.get(skill, "No course found") for skill in missing_skills}
    return suggested_courses

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
        # Extract text
        resume_text = extract_text_from_pdf(resume_file) if resume_file.name.endswith(".pdf") else extract_text_from_docx(resume_file)
        job_desc_text = extract_text_from_docx(job_desc_file) if job_desc_file.name.endswith(".docx") else job_desc_file.read().decode("utf-8")

        # Compare and display results
        match_score = compare_texts(resume_text, job_desc_text)
        st.subheader("🔍 Match Analysis")
        st.write(f"**Matching Score:** `{match_score}%`")

        # Extract skills from resume and job description
        resume_skills = extract_skills(resume_text)
        job_desc_skills = extract_skills(job_desc_text)
        missing_skills = list(set(job_desc_skills) - set(resume_skills))

        # Display Extracted Skills
        st.subheader("📌 Extracted Skills")
        st.write("✅ **Skills in Resume:**", ", ".join(resume_skills) if resume_skills else "No skills detected.")
        st.write("📌 **Skills in Job Description:**", ", ".join(job_desc_skills) if job_desc_skills else "No skills detected.")

        # Highlight Missing Skills
        if missing_skills:
            st.warning(f"⚠️ Missing Skills: {', '.join(missing_skills)}")
            
            # Suggest Courses
            st.subheader("🎓 Course Recommendations for Missing Skills")
            course_suggestions = suggest_courses(missing_skills)
            for skill, link in course_suggestions.items():
                st.write(f"- **{skill.title()}**: [Learn Here]({link})")
        else:
            st.success("✅ Your resume contains all required skills!")

        # Visualization Section
        st.subheader("📊 Match Score Breakdown")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.pie([match_score, 100 - match_score], labels=["Matched", "Not Matched"], autopct="%1.1f%%", colors=["#4CAF50", "#FF5733"], startangle=90)
        ax.set_title("Resume vs. Job Description Match")
        st.pyplot(fig)

        # Suggestions based on score
        st.subheader("📌 Recommendations")
        if match_score >= 80:
            st.success("✅ Your resume is a great match for this job! Keep it up.")
        elif 50 <= match_score < 80:
            st.warning("⚠️ Moderate match. Consider adding relevant skills and keywords.")
        else:
            st.error("❌ Your resume does not match well. Update it with relevant skills.")

        st.markdown("### 🔹 Optimization Tips:") 
        st.markdown("✅ Include key skills mentioned in the job description.")
        st.markdown("✅ Use action words and quantified achievements.")
        st.markdown
