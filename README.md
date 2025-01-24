# Resume ATS Scanner

A simple web application built with Streamlit to calculate an ATS (Applicant Tracking System) score by comparing resumes against job descriptions. The tool helps identify missing skills and optimize resumes for specific job roles.

## Features
- **Multi-format Support**: Upload resumes and job descriptions in PDF, DOCX, images, or plain text.
- **Skill Extraction**: Extracts relevant skills using SpaCy NLP.
- **ATS Score**: Combines cosine similarity and skill match metrics.

## How to Run
1. Clone the repository and navigate to the project directory:
   ```bash
   git clone https://github.com/Ritampatra76/resume-ats-scanner.git
   cd resume-ats-scanner
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the application:
   ```bash
   streamlit run app.py
   ```
4. Open the app in your browser at `http://localhost:8501`.

## License
Licensed under the [MIT License](LICENSE).

