import streamlit as st
import pdfplumber
import docx
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import textstat
import language_tool_python
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load NLP model
nlp = spacy.load("en_core_web_sm")
language_tool = language_tool_python.LanguageTool("en-US")

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
    return round(similarity_score * 100, 2)

# Function to extract key skills using NLP
def extract_keywords(text):
    doc = nlp(text)
    keywords = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"] and len(token.text) > 3]
    return list(set(keywords))[:10]

# Function to check grammar & readability
def grammar_readability_check(text):
    errors = len(language_tool.check(text))
    readability = textstat.flesch_reading_ease(text)
    return errors, readability

# Streamlit UI
st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

st.title("\ud83d\udcdd AI Resume Analyzer")
st.markdown("#### Upload your resume and job description to analyze compatibility.")

# Sidebar Upload Section
st.sidebar.header("\ud83d\udcc2 Upload Files")
resume_file = st.sidebar.file_uploader("Upload Resume (PDF/DOCX)", type=["pdf", "docx"])
job_desc_file = st.sidebar.file_uploader("Upload Job Description (TXT/DOCX)", type=["txt", "docx"])

if resume_file and job_desc_file:
    with st.spinner("\ud83d\udd0d Processing... Please wait"):
        resume_text = extract_text_from_pdf(resume_file) if resume_file.name.endswith(".pdf") else extract_text_from_docx(resume_file)
        job_desc_text = extract_text_from_docx(job_desc_file) if job_desc_file.name.endswith(".docx") else job_desc_file.read().decode("utf-8")
        
        resume_keywords = extract_keywords(resume_text)
        job_desc_keywords = extract_keywords(job_desc_text)
        match_score = compare_texts(resume_text, job_desc_text)
        grammar_errors, readability_score = grammar_readability_check(resume_text)
        
        st.subheader("\ud83d\udccc Resume Insights")
        st.write(f"**Top Skills Identified:** {', '.join(resume_keywords)}")
        st.write(f"**Readability Score:** {readability_score:.2f} (Higher is better)")
        st.write(f"**Grammar Errors:** {grammar_errors}")
        
        # Word Cloud
        st.subheader("\ud83c\udf1f Resume Word Cloud")
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(resume_text)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)
        
        # Visualization
        st.subheader("\ud83d\udcca Match Score Breakdown")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].pie([match_score, 100 - match_score], labels=["Match", "Mismatch"], autopct='%1.1f%%', colors=["skyblue", "lightgray"])
        axes[0].set_title("Resume vs. Job Description Match")
        sns.barplot(x=["Match Score"], y=[match_score], ax=axes[1], palette="Blues")
        axes[1].set_ylim(0, 100)
        axes[1].set_ylabel("Percentage")
        axes[1].set_title("Resume Match Percentage")
        st.pyplot(fig)
        
        # Recommendations
        st.subheader("\ud83d\udccc Recommendations")
        missing_skills = list(set(job_desc_keywords) - set(resume_keywords))
        
        if match_score >= 80:
            st.success("\u2705 Your resume is a great match for this job! Keep it up.")
        elif 50 <= match_score < 80:
            st.warning(f"\u26a0\ufe0f Moderate match. Consider adding relevant skills: {', '.join(missing_skills)}.")
        else:
            st.error(f"\u274c Your resume does not match well. Add missing skills: {', '.join(missing_skills)}.")
        
        # Download Report
        st.subheader("\ud83d\udcdd Download Report")
        report = f"""
        Resume Match Score: {match_score}%
        Readability Score: {readability_score:.2f}
        Grammar Errors: {grammar_errors}
        Missing Skills: {', '.join(missing_skills)}
        """
        st.download_button(label="Download Report as TXT", data=report, file_name="Resume_Analysis.txt")
        
        # AI Chatbot Advice
        st.subheader("\ud83e\udd16 AI Chatbot Advice")
        user_query = st.text_input("Ask for resume improvement suggestions:")
        if user_query:
            st.write("\ud83e\udd16 AI Suggestion: Focus on highlighting measurable achievements and using more action verbs in your experience section!")
