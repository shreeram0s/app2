import streamlit as st
import pdfplumber
import docx
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import textstat
import wordcloud
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from transformers import pipeline

# Ensure NLTK resources are available before using word_tokenize
nltk.download('punkt')

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    return text

# Function to extract text from DOCX
def extract_text_from_docx(docx_file):
    doc = docx.Document(docx_file)
    return "\n".join([para.text for para in doc.paragraphs])

# Function to extract keywords
def extract_keywords(text):
    words = word_tokenize(text)  # Now punkt is available
    words = [word.lower() for word in words if word.isalnum()]  # Remove punctuation
    return words

# Streamlit UI
st.title("AI Resume Analyzer")

uploaded_file = st.file_uploader("Upload Resume (PDF/DOCX)", type=["pdf", "docx"])

if uploaded_file:
    file_extension = uploaded_file.name.split(".")[-1]
    
    if file_extension == "pdf":
        resume_text = extract_text_from_pdf(uploaded_file)
    elif file_extension == "docx":
        resume_text = extract_text_from_docx(uploaded_file)
    else:
        st.error("Unsupported file format.")
        resume_text = ""

    if resume_text:
        st.subheader("Extracted Resume Text")
        st.text_area("Resume Content", resume_text, height=200)

        # Extract and display keywords
        resume_keywords = extract_keywords(resume_text)
        st.subheader("Resume Keywords")
        st.write(resume_keywords)
