# AI Resume Screening System

An AI-powered resume screening application that analyzes multiple resumes against a job description, identifies relevant skills, calculates matching scores, and ranks candidates to support the recruitment process.

## Project Overview

Recruiters often receive a large number of resumes for a single job opening. Manually reviewing every resume can be time-consuming and may lead to inconsistent screening.

The AI Resume Screening System helps simplify the initial screening process by extracting resume text, identifying relevant skills, comparing resumes with a job description, and displaying explainable matching scores through a professional web dashboard.

## Features

- **Job Description Input:** Enter the requirements and skills for a job role.
- **Multiple Resume Upload:** Upload multiple resumes in PDF and DOCX formats.
- **Resume Text Extraction:** Extract readable text from uploaded documents.
- **Skill Identification:** Identify relevant technical and soft skills using predefined skill aliases.
- **NLP-Based Similarity:** Use TF-IDF vectorization and cosine similarity to compare job descriptions and resumes.
- **Explainable Matching Scores:** Combine skill coverage and NLP similarity into a final matching score.
- **Candidate Ranking:** Rank candidates based on their calculated matching scores.
- **Skill Gap Analysis:** Display matched skills and missing job-related skills.
- **Candidate Information Extraction:** Attempt to extract contact information, education-related terms, and experience indicators.
- **CSV Export:** Download screening results for further analysis.
- **Interactive Dashboard:** View candidate rankings, score cards, and detailed screening results.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Application development |
| Streamlit | Interactive web application |
| Scikit-learn | TF-IDF vectorization and cosine similarity |
| PyMuPDF | PDF text extraction |
| python-docx | DOCX text extraction |
| Pandas | Tabular data processing and CSV export |
| Regular Expressions | Skill and basic candidate information extraction |

## Matching Methodology

The system combines two components:

1. **Skill Coverage:** Measures the proportion of skills detected in the job description that are also detected in a resume.
2. **NLP Similarity:** Uses TF-IDF vectors and cosine similarity to measure textual similarity between the job description and resume.

The final matching score is calculated as:

**Final Match Score = (Skill Coverage × Skill Weight) + (NLP Similarity × NLP Weight)**

The weights are expressed as proportions. For example, with a 60% skill weight and a 40% NLP weight:

Final Match Score = (Skill Coverage × 0.60) + (NLP Similarity × 0.40)

The skill and NLP weights can be adjusted in the application.

## Project Structure

```text
AI_Resume_Screening_System/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation and Setup

### Prerequisites

- Python 3.10 or a compatible Python version
- pip package manager
- Git
- Visual Studio Code or another Python editor

### 1. Clone the repository

Replace `YOUR-USERNAME` with your GitHub username.

```bash
git clone https://github.com/YOUR-USERNAME/AI-Resume-Screening-System.git
```

### 2. Open the project folder

```bash
cd AI-Resume-Screening-System
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can instead run the Python commands using the virtual environment's Python executable.

### 5. Install the dependencies

```bash
python -m pip install -r requirements.txt
```

### 6. Run the application

```bash
python -m streamlit run app.py
```

The application will open in your browser, usually at:

```text
http://localhost:8501
```

## How to Use

1. Open the AI Resume Screening System.
2. Enter the job description in the text area.
3. Upload one or more PDF or DOCX resumes.
4. Adjust the skill matching weight if required.
5. Click **Analyze and Rank Candidates**.
6. Review the candidate rankings and matching scores.
7. Inspect matched skills, missing skills, and extracted resume information.
8. Download the screening report as a CSV file.

## Limitations

- Skill identification depends on the predefined skill dictionary and its aliases.
- Scanned PDFs may not produce readable text because OCR is not currently included.
- Education and experience extraction is keyword-based and may not accurately identify qualifications or calculate years of experience.
- TF-IDF and cosine similarity measure textual alignment; they do not establish a candidate's true ability or suitability.
- The matching score has not been validated as a predictor of job performance.

## Future Enhancements

- Integrate OCR for scanned resumes.
- Improve education, experience, and qualification extraction.
- Add semantic similarity using transformer-based language models.
- Support more document formats.
- Add visual charts and candidate comparison features.
- Improve privacy controls and secure handling of candidate data.
- Evaluate the matching algorithm using a representative, appropriately labeled dataset.

## Ethical Considerations

This project is intended to assist with initial resume screening. Its scores should not be treated as definitive hiring decisions. Recruiters should review original resumes, verify qualifications, and assess candidates fairly. Avoid uploading personal or confidential resumes without appropriate authorization.

## Project Purpose

Developed as an academic project to demonstrate Python programming, natural language processing, text extraction, skill analysis, and interactive dashboard development.