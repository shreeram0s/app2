import streamlit as st
import pdfplumber
import docx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup
import requests

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

# Function to extract top keywords from text
def extract_keywords(text, top_n=10):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=top_n)
    tfidf_matrix = vectorizer.fit_transform([text])
    return vectorizer.get_feature_names_out()

# Function to compare resume and job description
def compare_texts(resume_text, job_desc_text):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_desc_text])
    similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    return round(similarity_score * 100, 2)  # Convert to percentage

# Function to find missing skills in resume
def find_missing_skills(resume_keywords, job_keywords):
    return list(set(job_keywords) - set(resume_keywords))

# Function to get online courses for missing skills
def get_courses(skill):
    search_url = f"https://www.udemy.com/courses/search/?q={skill.replace(' ', '+')}"
    response = requests.get(search_url)
    soup = BeautifulSoup(response.text, "html.parser")
    course_links = []
    
    # Scraping Udemy course links (Basic Implementation)
    for link in soup.find_all("a", class_="udlite-custom-focus-visible", href=True):
        course_links.append("https://www.udemy.com" + link["href"])
        if len(course_links) >= 3:  # Get only top 3 courses
            break
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
        # Extract text
        resume_text = extract_text_from_pdf(resume_file) if resume_file.name.endswith(".pdf") else extract_text_from_docx(resume_file)
        job_desc_text = extract_text_from_docx(job_desc_file) if job_desc_file.name.endswith(".docx") else job_desc_file.read().decode("utf-8")

        # Compare and display results
        match_score = compare_texts(resume_text, job_desc_text)
        st.subheader("🔍 Match Analysis")
        st.write(f"**Matching Score:** `{match_score}%`")

        # Extract keywords
        resume_keywords = extract_keywords(resume_text)
        job_keywords = extract_keywords(job_desc_text)
        missing_skills = find_missing_skills(resume_keywords, job_keywords)

        # Visualization - Match Score Breakdown
        st.subheader("📊 Match Score Breakdown")

        # Bar Chart - Match Score
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(["Match Score"], [match_score], color="skyblue")
        ax.set_ylim(0, 100)
        ax.set_ylabel("Percentage")
        ax.set_title("Resume vs. Job Description Match")
        st.pyplot(fig)

        # Pie Chart - Resume vs. JD focus
        labels = ['Resume Content', 'Job Description Focus']
        sizes = [len(resume_text.split()), len(job_desc_text.split())]
        colors = ['lightblue', 'lightcoral']
        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        ax.set_title("Resume vs. Job Description Focus")
        st.pyplot(fig)

        # Word Cloud - Resume & Job Description
        st.subheader("📌 Key Terms in Resume & Job Description")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Resume Keywords")
            wordcloud_resume = WordCloud(width=400, height=200, background_color='black').generate(resume_text)
            fig, ax = plt.subplots()
            ax.imshow(wordcloud_resume, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)

        with col2:
            st.markdown("#### Job Description Keywords")
            wordcloud_job = WordCloud(width=400, height=200, background_color='black').generate(job_desc_text)
            fig, ax = plt.subplots()
            ax.imshow(wordcloud_job, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)

        # Missing Skills & Course Recommendations
        st.subheader("📌 Missing Skills & Course Recommendations")
        if missing_skills:
            st.warning(f"🚀 You are missing these key skills: {', '.join(missing_skills)}")
            
            # Fetch Courses
            for skill in missing_skills[:3]:  # Limit to 3 skills
                st.markdown(f"**🔹 Courses for {skill}:**")
                courses = get_courses(skill)
                for course in courses:
                    st.markdown(f"- [Course Link]({course})")
        else:
            st.success("✅ Your resume covers all major skills from the job description!")

        # Suggestions based on match score
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
