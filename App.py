import streamlit as st
import re
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
import pdfplumber
import docx2txt

# Function to extract text from files
def extract_text(file):
    if file.type == "application/pdf":
        text = ""
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        return text
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return docx2txt.process(file)
    elif file.type == "text/plain":
        return file.read().decode("utf-8")
    else:
        return ""

# Function to tokenize text without NLTK
def tokenize_text(text):
    return re.findall(r'\b[a-zA-Z]+\b', text.lower())  # Extract words only

# Function to extract top keywords (skills) using TF-IDF
def extract_skills(resume_text, job_desc_text):
    vectorizer = TfidfVectorizer(stop_words="english")
    corpus = [resume_text, job_desc_text]
    X = vectorizer.fit_transform(corpus)
    
    # Get feature names (words) and their scores
    feature_names = vectorizer.get_feature_names_out()
    scores = X.toarray()
    
    resume_keywords = set([feature_names[i] for i in scores[0].argsort()[-15:]])  # Top 15 words in resume
    job_keywords = set([feature_names[i] for i in scores[1].argsort()[-15:]])  # Top 15 words in job desc
    
    missing_skills = job_keywords - resume_keywords  # Skills required but not in resume
    return list(missing_skills)

# Function to fetch online courses dynamically using Google
def fetch_course_links(skill):
    search_query = f"{skill} online course site:coursera.org OR site:udemy.com OR site:edx.org"
    url = f"https://www.google.com/search?q={search_query}"
    
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    
    links = []
    for g in soup.find_all("a", href=True):
        if "url?q=" in g["href"]:
            actual_link = g["href"].split("url?q=")[1].split("&")[0]
            if "coursera" in actual_link or "udemy" in actual_link or "edx" in actual_link:
                links.append(actual_link)
        if len(links) == 3:  # Fetch top 3 courses
            break
    return links

# Streamlit UI
st.title("AI Resume Analyzer")
resume_file = st.file_uploader("Upload your Resume (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
job_file = st.file_uploader("Upload Job Description (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

if resume_file and job_file:
    resume_text = extract_text(resume_file)
    job_desc_text = extract_text(job_file)
    
    missing_skills = extract_skills(resume_text, job_desc_text)
    
    if missing_skills:
        st.subheader("📌 Missing Skills Detected:")
        for skill in missing_skills:
            st.write(f"✅ {skill}")

        st.subheader("📚 Suggested Online Courses:")
        for skill in missing_skills:
            courses = fetch_course_links(skill)
            if courses:
                st.write(f"🔹 **{skill.capitalize()}**:")
                for course in courses:
                    st.markdown(f"[{course}]({course})")
            else:
                st.write(f"❌ No courses found for {skill}")
    else:
        st.success("Your resume matches all required skills!")
