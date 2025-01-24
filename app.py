import streamlit as st
import os
from PyPDF2 import PdfReader
from pytesseract import image_to_string
from pdf2image import convert_from_path
from PIL import Image
from docx import Document
from unstructured.partition.auto import partition
import spacy
from sentence_transformers import SentenceTransformer, util

# Load NLP models
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Text extraction functions
def extract_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def extract_image_pdf_text(pdf_path):
    images = convert_from_path(pdf_path)
    text = ""
    for image in images:
        text += image_to_string(image)
    return text

def extract_image_text(image_path):
    text = image_to_string(Image.open(image_path))
    return text

def extract_word_text(docx_path):
    doc = Document(docx_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text
    return text

def extract_generic_text(file_path):
    elements = partition(filename=file_path)
    text = " ".join([str(el) for el in elements])
    return text

def extract_text(file_path):
    if not os.path.exists(file_path):
        return "File not found. Please check the path."

    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()

    try:
        if file_extension == ".pdf":
            try:
                return extract_pdf_text(file_path)
            except Exception:
                return extract_image_pdf_text(file_path)

        elif file_extension in [".jpg", ".jpeg", ".png"]:
            return extract_image_text(file_path)

        elif file_extension == ".docx":
            return extract_word_text(file_path)

        else:
            return extract_generic_text(file_path)

    except Exception as e:
        return f"Error processing file: {str(e)}"

# Skill extraction function
def extract_skills(text):
    doc = nlp(text)
    skills = set()

    # Extract skills as named entities labeled "SKILL" or significant nouns
    for ent in doc.ents:
        if ent.label_ == "SKILL":
            skills.add(ent.text.lower())

    for token in doc:
        if token.pos_ in {"NOUN"} and not token.is_stop:
            skills.add(token.text.lower())

    return list(skills)

# ATS score calculation
def calculate_ats_score(resume_text, job_text):
    required_skills = extract_skills(job_text)
    resume_skills = extract_skills(resume_text)
    missing_skills = [skill for skill in required_skills if skill not in resume_skills]

    resume_embedding = model.encode(resume_text, convert_to_tensor=True)
    job_embedding = model.encode(job_text, convert_to_tensor=True)
    similarity_score = util.cos_sim(resume_embedding, job_embedding)[0][0].item() * 100

    skill_match_score = ((len(required_skills) - len(missing_skills)) / len(required_skills)) * 100 if required_skills else 0
    final_ats_score = 0.7 * similarity_score + 0.3 * skill_match_score

    return final_ats_score, similarity_score, skill_match_score, missing_skills

# Streamlit UI
st.markdown(
    """
    <h1 style="text-align: center; font-size: 44px;">Resume ATS Scanner</h1>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Resume")
    resume_input = st.text_area("Enter the text or upload a file", key="resume_input", height=150)
    resume_file = st.file_uploader("Upload Resume File", type=["txt", "pdf", "docx", "jpg", "png"], key="resume_file")

with col2:
    st.subheader("Job Description")
    jd_input = st.text_area("Enter the text or upload a file", key="jd_input", height=150)
    jd_file = st.file_uploader("Upload Job Description File", type=["txt", "pdf", "docx", "jpg", "png"], key="jd_file")

if st.button("Calculate Your ATS Score", key="calculate-btn"):
    if resume_input or resume_file:
        if resume_file:
            resume_path = f"temp_resume{os.path.splitext(resume_file.name)[-1]}"
            with open(resume_path, "wb") as f:
                f.write(resume_file.getbuffer())
            resume_text = extract_text(resume_path)
        else:
            resume_text = resume_input

        if jd_input or jd_file:
            if jd_file:
                jd_path = f"temp_jd{os.path.splitext(jd_file.name)[-1]}"
                with open(jd_path, "wb") as f:
                    f.write(jd_file.getbuffer())
                job_text = extract_text(jd_path)
            else:
                job_text = jd_input

            ats_score, similarity_score, skill_match_score, missing_skills = calculate_ats_score(resume_text, job_text)

            st.markdown(f"""
                <h3 style="text-align: center;">ATS Score: {ats_score:.2f}%</h3>
                <p style="text-align: center;">Cosine Similarity: {similarity_score:.2f}%</p>
                <p style="text-align: center;">Skill Match Score: {skill_match_score:.2f}%</p>
            """, unsafe_allow_html=True)

            if missing_skills:
                st.markdown(f"""
                    <p style="text-align: center; color: red;">Missing Skills: {', '.join(missing_skills)}</p>
                    <p style="text-align: center; color: red;">Consider updating your resume to include the missing skills.</p>
                """, unsafe_allow_html=True)
            else:
                st.markdown("<p style=\"text-align: center; color: green;\">Your resume matches the job description perfectly!</p>", unsafe_allow_html=True)
        else:
            st.error("Please provide a job description!")
    else:
        st.error("Please provide a resume!")
