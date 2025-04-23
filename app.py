import streamlit as st
import google.generativeai as genai
import os
import PyPDF2 as pdf
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to extract text from uploaded PDF
def extract_pdf_text(uploaded_file):
    reader = pdf.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to get response from Gemini model
def get_ats_feedback(resume_text, jd_text):
    prompt = f""" 
    Act as an expert ATS (Application Tracking System) for the tech industry.
    Evaluate the resume below against the job description and return a JSON response with:
    - JD Match (as percentage),
    - List of Missing Keywords,
    - A brief but insightful Profile Summary.

    Resume: {resume_text}
    Job Description: {jd_text}

    Format:
    {{
        "JD Match": "85%",
        "MissingKeywords": ["Python", "Cloud Computing"],
        "Profile Summary": "Strong backend skills with exposure to databases and APIs..."
    }}
    """
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt)
    return response.text

# ---- Streamlit App ----
st.set_page_config(page_title="Verq ATS Evaluator", layout="centered")

# Brand header
st.markdown("""
    <div style="text-align: center;">
        <h1 style="font-size: 3em; color: #4B8BBE; margin-bottom: 0;">🚀 Verq</h1>
        <p style="font-size: 1.2em; color: #666;">Helping Computer Science students craft optimized, job-ready resumes for the tech industry's best opportunities.</p>
    </div>
    <br>
""", unsafe_allow_html=True)

st.markdown("##  ATS Resume Evaluator")
st.markdown("Upload your resume and job description to receive a tailored match percentage, keyword analysis, and improvement suggestions.")

# Input columns
with st.container():
    col1, col2 = st.columns(2)

    with col1:
        jd_input = st.text_area(" Job Description", height=300, placeholder="Paste the JD here...")

    with col2:
        uploaded_resume = st.file_uploader(" Upload Resume (PDF)", type=["pdf"])

# Submit button
if st.button(" Evaluate"):
    if uploaded_resume and jd_input.strip():
        with st.spinner("Analyzing Resume..."):
            resume_text = extract_pdf_text(uploaded_resume)
            ats_response = get_ats_feedback(resume_text, jd_input)

        st.markdown("---")
        st.markdown("###  ATS Evaluation Results")
        st.code(ats_response, language='json')
    else:
        st.warning("Please upload a resume and enter a job description.")
