import streamlit as st
import pdfplumber
import docx
import requests
import json
import matplotlib.pyplot as plt
import nltk
import re
import pandas as pd
from collections import Counter
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer

# Download necessary NLTK datasets
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")

# Function to extract text from resume and job description
def extract_text_from_file(uploaded_file):
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split(".")[-1]
        if file_extension == "pdf":
            with pdfplumber.open(uploaded_file) as pdf:
                return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        elif file_extension in ["docx", "doc"]:
            doc = docx.Document(uploaded_file)
            return "\n".join([para.text for para in doc.paragraphs])
        elif file_extension == "txt":
            return uploaded_file.read().decode("utf-8")
    return None

# Function to clean and preprocess text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
    return text

# Function to dynamically extract relevant skills using NLTK & TF-IDF
def extract_skills(text):
    text = clean_text(text)
    words = nltk.word_tokenize(text)
    tagged_words = nltk.pos_tag(words)  # Identify parts of speech
    skills = [word for word, tag in tagged_words if tag in ["NN", "NNS", "JJ"]]  # Extract nouns and adjectives
    
    # Use TF-IDF to identify most important words
    vectorizer = TfidfVectorizer(stop_words="english", max_features=20)
    tfidf_matrix = vectorizer.fit_transform([" ".join(skills)])
    extracted_skills = vectorizer.get_feature_names_out()

    return list(set(extracted_skills))

# Function to identify missing skills
def find_missing_skills(resume_text, job_desc_text):
    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_desc_text)

    missing_skills = list(set(job_skills) - set(resume_skills))
    return missing_skills

# Function to fetch online course links using SerpAPI
def fetch_course_links(skill):
    serp_api_key = "YOUR_SERPAPI_KEY"  # Replace with your SerpAPI key
    search_query = f"best online course for {skill}"
    
    params = {
        "engine": "google",
        "q": search_query,
        "api_key": serp_api_key
    }
    
    response = requests.get("https://serpapi.com/search", params=params)
    results = json.loads(response.text)

    links = []
    if "organic_results" in results:
        for result in results["organic_results"][:3]:  # Get top 3 links
            links.append(result["link"])
    
    return links if links else ["No resources found"]

# Function to generate a structured learning plan
def generate_learning_plan(missing_skills):
    learning_plan = {}
    for index, skill in enumerate(missing_skills, start=1):
        courses = fetch_course_links(skill)
        learning_plan[f"Week {index} - {skill}"] = courses
    return learning_plan

# Streamlit UI
st.title("📌 AI Resume Analyzer (Advanced NLP)")

st.sidebar.header("Upload Your Files")
uploaded_resume = st.sidebar.file_uploader("Upload Resume (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
uploaded_job_desc = st.sidebar.file_uploader("Upload Job Description (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

if uploaded_resume and uploaded_job_desc:
    resume_text = extract_text_from_file(uploaded_resume)
    job_desc_text = extract_text_from_file(uploaded_job_desc)

    if resume_text and job_desc_text:
        missing_skills = find_missing_skills(resume_text, job_desc_text)

        if missing_skills:
            st.subheader("🚀 Missing Skills Identified")
            st.write(missing_skills)

            # Generate Learning Plan
            learning_plan = generate_learning_plan(missing_skills)
            st.subheader("📚 Personalized Learning Schedule")

            for week, courses in learning_plan.items():
                st.write(f"**{week}**")
                for course in courses:
                    st.markdown(f"[🔗 {course}]({course})", unsafe_allow_html=True)

            # Visualizing Skill Gaps
            st.subheader("📊 Skill Gap Analysis")
            fig, ax = plt.subplots()
            ax.bar(missing_skills, range(len(missing_skills)), color="blue")
            ax.set_ylabel("Importance Level")
            ax.set_title("Skill Gap Analysis")
            plt.xticks(rotation=45)
            st.pyplot(fig)

        else:
            st.success("✅ Your resume matches all required job skills!")
