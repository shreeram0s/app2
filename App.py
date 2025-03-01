import streamlit as st
import pdfplumber
import docx
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import requests

# Load NLP model for skill extraction
nlp = spacy.load("en_core_web_sm")

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

# Function to extract skills from text
def extract_skills(text):
    doc = nlp(text)
    skills = {token.text.lower() for token in doc.ents if token.label_ == "SKILL"}
    return skills

# Function to compare resume and job description
def compare_texts(resume_text, job_desc_text):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_desc_text])
    similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    return round(similarity_score * 100, 2)

# Function to fetch courses for missing skills
def get_courses(skill):
    try:
        response = requests.get(f'https://www.udemy.com/api-2.0/courses/?search={skill}')
        if response.status_code == 200:
            courses = response.json()
            return [course['title'] for course in courses.get('results', [])[:3]]
    except:
        return []
    return []

# Streamlit UI
st.set_page_config(page_title="AI Resume Analyzer", layout="wide")
st.title("📄 AI Resume Analyzer")
st.markdown("#### Upload your resume and job description to analyze compatibility and skill gaps.")

# Sidebar Upload Section
st.sidebar.header("📂 Upload Files")
resume_file = st.sidebar.file_uploader("Upload Resume (PDF/DOCX)", type=["pdf", "docx"])
job_desc_file = st.sidebar.file_uploader("Upload Job Description (TXT/DOCX)", type=["txt", "docx"])

if resume_file and job_desc_file:
    with st.spinner("🔍 Processing... Please wait"):
        resume_text = extract_text_from_pdf(resume_file) if resume_file.name.endswith(".pdf") else extract_text_from_docx(resume_file)
        job_desc_text = extract_text_from_docx(job_desc_file) if job_desc_file.name.endswith(".docx") else job_desc_file.read().decode("utf-8")
        
        # Compare texts
        match_score = compare_texts(resume_text, job_desc_text)
        st.subheader("🔍 Match Analysis")
        st.write(f"**Matching Score:** `{match_score}%`")
        
        # Extract and compare skills
        resume_skills = extract_skills(resume_text)
        job_desc_skills = extract_skills(job_desc_text)
        missing_skills = job_desc_skills - resume_skills
        
        st.subheader("🛠️ Skill Comparison")
        st.write(f"✅ **Skills in Resume:** {', '.join(resume_skills) if resume_skills else 'No skills detected'}")
        st.write(f"🎯 **Required Skills:** {', '.join(job_desc_skills) if job_desc_skills else 'No skills detected'}")
        st.write(f"❌ **Missing Skills:** {', '.join(missing_skills) if missing_skills else 'None! Your resume is well-matched!'}")
        
        # Visualization
        st.subheader("📊 Match Score Breakdown")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.pie([match_score, 100 - match_score], labels=["Matched", "Not Matched"], autopct='%1.1f%%', colors=["#4CAF50", "#FF5733"], startangle=90)
        ax.set_title("Resume vs. Job Description Match")
        st.pyplot(fig)
        
        # Course recommendations for missing skills
        if missing_skills:
            st.subheader("📚 Suggested Courses for Missing Skills")
            for skill in missing_skills:
                courses = get_courses(skill)
                if courses:
                    st.write(f"### 🔹 {skill.capitalize()}:")
                    for course in courses:
                        st.write(f"- {course}")
                else:
                    st.write(f"No courses found for {skill}. Try searching manually!")
        
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
        st.markdown("✅ Align your experience with job requirements.")

st.sidebar.info("🔹 Make sure your resume is optimized for best results.")
