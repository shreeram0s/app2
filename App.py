import streamlit as st
import pandas as pd
import numpy as np
import PyPDF2
import io
from fpdf import FPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Function to extract text from uploaded PDF
def extract_text_from_pdf(uploaded_file):
    text = ""
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to compare resume and job description
def analyze_resume_vs_job(resume_text, job_desc_text):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_desc_text])
    similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return similarity_score, vectorizer

# Function to identify missing skills
def identify_missing_skills(resume_text, job_desc_text):
    resume_words = set(resume_text.lower().split())
    job_words = set(job_desc_text.lower().split())
    missing_skills = job_words - resume_words
    return list(missing_skills)

# Function to create a structured learning schedule
def create_learning_schedule(missing_skills):
    days_required = len(missing_skills)
    schedule = []
    
    for day in range(1, days_required + 1):
        schedule.append({
            "Day": day,
            "Time": "2 Hours",
            "Topic": missing_skills[day - 1],
            "Resources": f"https://www.google.com/search?q=learn+{missing_skills[day - 1]}"  # Auto Google Search
        })
    
    return pd.DataFrame(schedule)

# Function to generate a PDF for the learning schedule
def generate_pdf(schedule_df):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="Skill Learning Schedule", ln=True, align="C")
    pdf.ln(10)

    for index, row in schedule_df.iterrows():
        pdf.cell(200, 10, txt=f"Day {row['Day']} - {row['Topic']}", ln=True, align="L")
        pdf.cell(200, 10, txt=f"Time: {row['Time']}", ln=True, align="L")
        pdf.cell(200, 10, txt=f"Resource: {row['Resources']}", ln=True, align="L")
        pdf.ln(10)

    pdf_output = io.BytesIO()
    pdf.output(pdf_output)
    pdf_output.seek(0)

    return pdf_output

# Streamlit UI
st.title("AI Resume Analyzer 📄🤖")

# Upload Resume and Job Description
resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
job_desc_text = st.text_area("Paste Job Description Here", height=200)

if resume_file and job_desc_text:
    # Extract text from resume
    resume_text = extract_text_from_pdf(resume_file)
    
    # Show extracted content
    st.subheader("Extracted Resume Text")
    st.write(resume_text[:500] + "...")  # Display first 500 characters
    
    st.subheader("Job Description Preview")
    st.write(job_desc_text[:500] + "...")  # Display first 500 characters
    
    # Compare resume with job description
    similarity_score, vectorizer = analyze_resume_vs_job(resume_text, job_desc_text)
    
    st.subheader("Similarity Score")
    st.write(f"Your Resume matches **{similarity_score * 100:.2f}%** with the Job Description.")
    
    # Identify missing skills
    missing_skills = identify_missing_skills(resume_text, job_desc_text)
    
    st.subheader("Missing Skills")
    if missing_skills:
        st.write(", ".join(missing_skills))
    else:
        st.success("No missing skills found! 🎉")

    # Visualization
    st.subheader("Skill Match Visualization")
    labels = ["Matched", "Missing"]
    values = [len(set(resume_text.lower().split()) & set(job_desc_text.lower().split())), len(missing_skills)]
    
    fig, ax = plt.subplots()
    ax.pie(values, labels=labels, autopct="%1.1f%%", colors=["green", "red"])
    ax.set_title("Resume vs Job Skills")
    st.pyplot(fig)

    # Generate Learning Schedule
    if missing_skills:
        st.subheader("Personalized Learning Plan 📅")
        schedule_df = create_learning_schedule(missing_skills)
        st.table(schedule_df)
        
        # Generate PDF
        pdf_file = generate_pdf(schedule_df)
        st.download_button(label="Download Learning Plan as PDF", 
                           data=pdf_file, 
                           file_name="learning_schedule.pdf", 
                           mime="application/pdf")
