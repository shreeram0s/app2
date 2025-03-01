import streamlit as st
import pandas as pd
import io
import PyPDF2
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fpdf import FPDF

# Predefined learning resources for missing skills
learning_resources = {
    "SQL": [
        ("Day 1", "Introduction to Databases & SQL Basics", "https://www.coursera.org/learn/sql-for-data-science"),
        ("Day 2", "SELECT, WHERE, and Filtering Data", "https://www.w3schools.com/sql/"),
        ("Day 3", "Aggregation & Joins", "https://www.kaggle.com/learn/advanced-sql"),
        ("Day 4", "Writing Complex Queries", "https://www.udemy.com/course/sql-intermediate/"),
        ("Day 5", "Hands-on Practice: Solve 10 SQL Problems", "https://leetcode.com/problemset/database/"),
        ("Day 6", "SQL Project: Building a Mini Database", "https://www.datacamp.com/"),
        ("Day 7", "Revision & Mock Test", "https://www.testdome.com/")
    ],
    "Power BI": [
        ("Day 1", "Introduction to Power BI", "https://www.udemy.com/course/microsoft-power-bi/"),
        ("Day 2", "Connecting Data Sources", "https://www.youtube.com/watch?v=JgR29b8L7N4"),
        ("Day 3", "Creating Interactive Dashboards", "https://www.coursera.org/learn/power-bi-dashboards"),
        ("Day 4", "Hands-on Practice: Build a Sales Dashboard", "https://www.datacamp.com/"),
        ("Day 5", "Advanced Visualizations & DAX", "https://www.sqlbi.com/"),
        ("Day 6", "Final Power BI Project", "https://www.kaggle.com/"),
        ("Day 7", "Review & Feedback", "https://www.linkedin.com/")
    ],
    "Python": [
        ("Day 1", "Python Basics: Variables, Data Types", "https://www.w3schools.com/python/"),
        ("Day 2", "Loops and Conditionals", "https://www.udemy.com/course/python-for-beginners-learn-programming-from-scratch/"),
        ("Day 3", "Functions & Modules", "https://www.coursera.org/learn/python"),
        ("Day 4", "Object-Oriented Programming", "https://realpython.com/python-oop/"),
        ("Day 5", "Data Analysis with Pandas", "https://www.datacamp.com/courses/pandas-foundations"),
        ("Day 6", "Python Mini-Project", "https://www.kaggle.com/"),
        ("Day 7", "Final Review & Mock Tests", "https://www.hackerrank.com/domains/tutorials/10-days-of-python")
    ]
}

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + " "
    return text

# Function to extract skills using regex
def extract_skills(text):
    skills = {"SQL", "Power BI", "Python"}  # Expand this with more skills
    found_skills = {skill for skill in skills if re.search(rf"\b{skill}\b", text, re.IGNORECASE)}
    return found_skills

# Function to identify missing skills
def find_missing_skills(resume_text, job_desc_text):
    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_desc_text)
    missing_skills = job_skills - resume_skills
    return list(missing_skills), list(resume_skills), list(job_skills)

# Function to generate a structured learning plan
def generate_learning_plan(missing_skills):
    schedule = []
    for skill in missing_skills:
        if skill in learning_resources:
            schedule.extend(learning_resources[skill])
    return pd.DataFrame(schedule, columns=["Day", "Topic", "Resource Link"])

# Function to generate a PDF for downloading
def generate_pdf(schedule_df):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="Skill Learning Schedule", ln=True, align="C")
    pdf.ln(10)

    for _, row in schedule_df.iterrows():
        pdf.cell(200, 10, txt=f"Day {row['Day']} - {row['Topic']}", ln=True, align="L")
        pdf.cell(200, 10, txt=f"Resource: {row['Resource Link']}", ln=True, align="L")
        pdf.ln(10)

    pdf_output = io.BytesIO()
    pdf.output(pdf_output, 'F')
    pdf_output.seek(0)

    return pdf_output

# Streamlit UI
st.title("📄 AI Resume Analyzer - Skill Gap Learning Plan")
st.subheader("📌 Upload your resume and job description to identify skill gaps!")

# Upload resume
resume_file = st.file_uploader("Upload your Resume (PDF)", type=["pdf"])
job_desc = st.text_area("Paste the Job Description")

if resume_file and job_desc:
    with st.spinner("Processing..."):
        resume_text = extract_text_from_pdf(resume_file)
        missing_skills, resume_skills, job_skills = find_missing_skills(resume_text, job_desc)

        # Display brief descriptions of resume and job description
        st.subheader("🔎 Resume Summary")
        st.write(resume_text[:500] + "...")  # Display first 500 characters

        st.subheader("📄 Job Description Overview")
        st.write(job_desc[:500] + "...")  # Display first 500 characters

        # Skill matching visualization
        st.subheader("📊 Skill Matching Visualization")
        skill_labels = ["Matched Skills", "Missing Skills"]
        skill_counts = [len(resume_skills), len(missing_skills)]

        fig, ax = plt.subplots()
        ax.bar(skill_labels, skill_counts, color=["green", "red"])
        ax.set_ylabel("Count")
        ax.set_title("Resume vs. Job Description Skills")
        st.pyplot(fig)

        if missing_skills:
            st.success(f"🔍 Missing Skills Identified: {', '.join(missing_skills)}")

            # Recommendations based on missing skills
            st.subheader("📌 Recommendations")
            st.write("To improve your resume, consider learning these missing skills through the suggested courses below.")

            # Generate learning plan
            schedule_df = generate_learning_plan(missing_skills)
            st.subheader("📅 Personalized Learning Schedule")
            st.dataframe(schedule_df)

            # Generate PDF
            pdf_file = generate_pdf(schedule_df)
            st.download_button(label="📥 Download Learning Plan as PDF",
                               data=pdf_file,
                               file_name="learning_schedule.pdf",
                               mime="application/pdf")
        else:
            st.success("✅ No missing skills detected! Your resume is well-matched.")
