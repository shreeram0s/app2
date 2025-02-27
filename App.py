import streamlit as st
import pandas as pd
import numpy as np
import docx2txt
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load NLP model
nlp = spacy.load('en_core_web_sm')

def extract_text_from_file(uploaded_file):
    if uploaded_file is not None:
        return docx2txt.process(uploaded_file)
    return ""

def calculate_similarity(resume_text, job_desc_text):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_desc_text])
    similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return round(similarity_score[0][0] * 100, 2)

def extract_skills(text):
    doc = nlp(text)
    skills = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]]
    return list(set(skills))

def plot_wordcloud(text, title):
    wordcloud = WordCloud(width=400, height=200, background_color='white').generate(text)
    plt.figure(figsize=(5,3))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    st.pyplot(plt)

def plot_skill_comparison(resume_skills, job_skills):
    skills = list(set(resume_skills + job_skills))
    resume_counts = [1 if skill in resume_skills else 0 for skill in skills]
    job_counts = [1 if skill in job_skills else 0 for skill in skills]
    df = pd.DataFrame({'Skills': skills, 'Resume': resume_counts, 'Job Description': job_counts})
    df.set_index('Skills', inplace=True)
    df.plot(kind='bar', figsize=(8,5))
    plt.title('Skill Comparison')
    plt.xlabel('Skills')
    plt.ylabel('Presence')
    st.pyplot(plt)

def main():
    st.set_page_config(page_title="AI Resume Analyzer", layout="wide")
    st.title("📄 AI Resume Analyzer")
    st.sidebar.header("Upload Your Files")
    resume_file = st.sidebar.file_uploader("Upload Resume (DOCX)", type=['docx'])
    job_desc_file = st.sidebar.file_uploader("Upload Job Description (TXT)", type=['txt'])

    if resume_file and job_desc_file:
        resume_text = extract_text_from_file(resume_file)
        job_desc_text = extract_text_from_file(job_desc_file)
        
        similarity = calculate_similarity(resume_text, job_desc_text)
        resume_skills = extract_skills(resume_text)
        job_skills = extract_skills(job_desc_text)
        
        st.metric(label="Resume Match Score", value=f"{similarity}%")
        st.progress(similarity / 100)
        
        col1, col2 = st.columns(2)
        with col1:
            plot_wordcloud(resume_text, "Resume Word Cloud")
        with col2:
            plot_wordcloud(job_desc_text, "Job Description Word Cloud")
        
        st.subheader("📊 Skill Comparison")
        plot_skill_comparison(resume_skills, job_skills)
        
        missing_skills = set(job_skills) - set(resume_skills)
        st.subheader("🛠️ Skills to Improve")
        st.write(f"Consider adding these skills to match the job better: {', '.join(missing_skills) if missing_skills else 'None'}")
    
if __name__ == "__main__":
    main()
