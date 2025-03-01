import streamlit as st
import pdfplumber
import docx
import matplotlib.pyplot as plt
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

st.title("\U0001F4C4 AI Resume Analyzer")  # 📄 Unicode emoji
st.markdown("#### Upload your resume and job description to analyze compatibility.")

# Sidebar Upload Section
st.sidebar.header("\U0001F4C2 Upload Files")  # 📂 Folder emoji
resume_file = st.sidebar.file_uploader("Upload Resume (PDF/DOCX)", type=["pdf", "docx"])
job_desc_file = st.sidebar.file_uploader("Upload Job Description (TXT/DOCX)", type=["txt", "docx"])

if resume_file and job_desc_file:
    with st.spinner("\ud83d\udd0d Processing... Please wait"):
        # Extract text
        resume_text = extract_text_from_pdf(resume_file) if resume_file.name.endswith(".pdf") else extract_text_from_docx(resume_file)
        job_desc_text = extract_text_from_docx(job_desc_file) if job_desc_file.name.endswith(".docx") else job_desc_file.read().decode("utf-8")
        
        # Compare and display results
        match_score = compare_texts(resume_text, job_desc_text)
        st.subheader("\ud83d\udd0d Match Analysis")
        st.write(f"**Matching Score:** `{match_score}%`")
        
        # Detailed Information on Resume vs. Job Description
        st.subheader("\ud83d\udcdd Resume Summary vs. Job Description Summary")
        col1, col2 = st.columns(2)
        col1.markdown("**Resume Extract:**")
        col1.write(resume_text[:500] + "...")  # Displaying only first 500 characters
        col2.markdown("**Job Description Extract:**")
        col2.write(job_desc_text[:500] + "...")
        
        # Visualization Section
        st.subheader("\ud83d\udcca Match Score Breakdown")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.pie([match_score, 100 - match_score], labels=["Matched", "Not Matched"], autopct='%1.1f%%', colors=["#4CAF50", "#FF5733"], startangle=90)
        ax.set_title("Resume vs. Job Description Match")
        st.pyplot(fig)
        
        # Suggestions based on score
        st.subheader("\ud83d\udccc Recommendations")
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
