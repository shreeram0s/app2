import streamlit as st
import pdfplumber
import docx
import requests
import json
import matplotlib.pyplot as plt
import spacy
import pandas as pd
from collections import Counter
import tiktoken  # For better tokenization
from bs4 import BeautifulSoup

# Load Spacy NLP model for Named Entity Recognition (NER)
nlp = spacy.load("en_core_web_sm")

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

# Function to extract relevant skills using NLP
def extract_skills(text):
    doc = nlp(text)
    skills = []
    
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PRODUCT", "SKILL"]:
            skills.append(ent.text.lower())
    
    # Use Tiktoken tokenizer to refine the extracted skills
    encoding = tiktoken.get_encoding("cl100k_base")
    tokenized_skills = [word for word in encoding.encode(" ".join(skills)) if word.isalpha()]
    
    return list(set(tokenized_skills))

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
st.title("📌 Advanced AI Resume Analyzer")

st.sidebar.header("Upload Your Files")
uploaded_resume = st.sidebar.file_uploader("Upload Resume (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
uploaded_job_desc = st.sidebar.file_uploader("Upload Job Description (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

if uploaded_resume and uploaded_job_desc:
    resume_text = extract_text_from_file(uploaded_resume)
    job_desc_text = extract_text_from_file(uploaded_job_desc)

    if resume_text and job_desc_text:
        st.subheader("📄 Resume & Job Description Summary")
        st.write(f"**Resume Summary:** {resume_text[:500]}...")
        st.write(f"**Job Description Summary:** {job_desc_text[:500]}...")

        missing_skills = find_missing_skills(resume_text, job_desc_text)
        
        if missing_skills:
            st.subheader("🚀 Skills You Need to Learn")
            st.warning(f"You need to learn: {', '.join(missing_skills)}")

            # Generate Learning Plan
            learning_plan = generate_learning_plan(missing_skills)

            st.subheader("📅 Personalized Learning Schedule")
            for week, courses in learning_plan.items():
                st.markdown(f"### {week}")
                for course in courses:
                    if course == "No resources found":
                        st.write("❌ No resources found")
                    else:
                        st.markdown(f"✅ [Course Link]({course})")

            # Skill Gap Visualization
            st.subheader("📊 Skill Gap Analysis")
            skill_counts = {"Possessed": len(extract_skills(resume_text)), "Missing": len(missing_skills)}
            plt.figure(figsize=(5, 3))
            plt.bar(skill_counts.keys(), skill_counts.values(), color=["green", "red"])
            plt.xlabel("Skills")
            plt.ylabel("Count")
            plt.title("Skill Gap Analysis")
            st.pyplot(plt)

        else:
            st.success("Your resume matches all required skills! ✅")
