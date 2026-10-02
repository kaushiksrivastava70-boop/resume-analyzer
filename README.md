# 📄 Resume Analyzer & ATS Optimizer

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests: Pytest](https://img.shields.io/badge/tests-pytest-brightgreen.svg)](https://docs.pytest.org/)

> A lightweight, client-side, privacy-first **Resume Analyzer & ATS Audit System** built with Python and Streamlit. Evaluates resumes across 5 key dimensions, extracts over 250+ technical and soft skills, compares qualifications against target Job Descriptions (JD), and generates instant PDF/text audit reports with prioritized recommendations.

---

## 🌟 Live Demo

🔗 **Demo URL:** [Deploy on Streamlit Community Cloud](https://share.streamlit.io/) *(Replace with your live Streamlit Cloud link after deploying)*

---

## 🚀 Key Features

- **📑 Multi-Format Parsing:** Drag-and-drop parsing for **PDF** and **DOCX** files, with intelligent detection for scanned or image-based PDFs.
- **⚡ 100-Point ATS Scoring Engine:**
  - **Contact Information (15 pts):** Email, phone, LinkedIn, GitHub, and portfolio links.
  - **Section Completeness (20 pts):** Summary, Experience, Education, Projects, Skills, and Certifications.
  - **Skills Detection & Breadth (25 pts):** Over 250+ categorized technical, cloud, data, web, and soft skills with regex boundary matching to prevent false positives.
  - **Quantified Impact & Action Verbs (20 pts):** Identifies percentages, metrics, dollar values, multipliers, and strong action verbs.
  - **Length & Readability (20 pts):** Flesch Reading Ease algorithm and word count evaluation for optimal ATS parsing.
- **🎯 Job Description Matcher:** Paste any job posting to calculate match percentage, view side-by-side matched vs. missing skills, and discover keyword gaps.
- **💡 Actionable Improvement Tips:** Prioritized recommendations (**High**, **Medium**, **Low**) ordered by hiring impact.
- **📊 Modern Interactive Visualizations:** Circular gauge chart and 5-axis radar chart built with Plotly.
- **🏢 Recruiter Mode (Batch Screening & Ranking):**
  - **In-Memory ZIP Processing:** Unpack and parse batches of PDF/DOCX resumes directly in memory.
  - **Deterministic TF-IDF & Cosine Similarity:** Semantic keyword and context alignment scored against the target Job Description.
  - **Dynamic Candidate Leaderboard:** Sortable, interactive candidate rankings with pass/fail threshold filtering.
  - **Batch Talent Analytics:** Top 5 missing skills across the applicant pool, score distribution histograms, and shortlisting metrics.
  - **Candidate Dossier Drilldown:** Instant candidate contact card, years of experience, matched skills, and extracted bullet points.
  - **Excel & CSV Export:** Download comprehensive candidate ranking reports as `.csv` or formatted `.xlsx`.
  - **1-Click Demo Batch:** Instant access to 5 realistic engineering candidate resumes for immediate demonstration.
- **📥 Downloadable Audit Reports:** Export your complete ATS evaluation as a **PDF** or structured **Text** file with one click.
- **🛡️ 100% Privacy-Preserving:** Resumes are processed completely in-memory. No databases, no tracking, and no external paid APIs required.
- **✨ One-Click Sample Resume:** Test all capabilities immediately without uploading your own document.

---

## 📸 Screenshots

*(Add screenshots of your deployed app here)*

| Candidate Mode | Recruiter Leaderboard |
| :---: | :---: |
| ![Dashboard Overview](https://raw.githubusercontent.com/placeholder/resume-analyzer-overview.png) | ![Recruiter Leaderboard](https://raw.githubusercontent.com/placeholder/recruiter-mode-leaderboard.png) |

---

## 🛠️ Tech Stack

- **Framework:** [Streamlit](https://streamlit.io/) (v1.32+)
- **Language:** Python 3.10+
- **Data & Tables:** [pandas](https://pandas.pydata.org/), [openpyxl](https://openpyxl.readthedocs.io/)
- **PDF Extraction:** [pypdf](https://pypdf.readthedocs.io/)
- **DOCX Extraction:** [python-docx](https://python-docx.readthedocs.io/)
- **Visualizations:** [Plotly](https://plotly.com/python/)
- **PDF Report Generation:** [ReportLab](https://www.reportlab.com/)
- **Testing:** [Pytest](https://docs.pytest.org/)

---

## 📂 Project Structure

```text
resume-analyzer/
├── app.py                  # Main Streamlit web application & UI (Candidate + Recruiter Modes)
├── analyzer.py             # Core candidate analysis engine & report generators
├── recruiter_mode.py       # Batch ZIP screening, TF-IDF cosine similarity & analytics
├── skills_db.py            # Categorized skills taxonomy & action verbs dictionary
├── sample_resume.txt       # Preloaded sample resume for 1-click testing
├── requirements.txt        # Production Python dependencies
├── .streamlit/
│   └── config.toml         # Custom modern theme configuration
├── .gitignore              # Git ignore rules for Python & Streamlit
├── README.md               # Project documentation
└── tests/
    ├── test_analyzer.py    # Automated unit tests for candidate analyzer
    └── test_recruiter_mode.py # Unit & integration tests for recruiter batch mode
```

---

## 💻 Local Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/resume-analyzer.git
cd resume-analyzer
```

### 2. Create and activate a virtual environment
```bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows (Command Prompt / PowerShell)
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run automated tests
```bash
pytest tests/test_analyzer.py -v
```

### 5. Launch the Streamlit web app
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## ☁️ Deployment Guide (Streamlit Community Cloud)

Deploying to **Streamlit Community Cloud** is completely free:

1. **Push your repository to GitHub** (see Git commands below).
2. Go to [share.streamlit.io](https://share.streamlit.io/) and sign in with your GitHub account.
3. Click **"New app"**.
4. Select your repository: `YOUR_USERNAME/resume-analyzer`.
5. Set the Main file path to: `app.py`.
6. Click **"Deploy!"**.
7. Your app will build, install dependencies from `requirements.txt`, and go live in ~1-2 minutes!

---

## 📤 Git Commands to Push to GitHub

Run these commands in your project terminal:

```bash
# Initialize git repository
git init

# Stage all files
git add .

# Create initial commit
git commit -m "feat: complete production-ready Resume Analyzer Streamlit app"

# Rename branch to main
git branch -M main

# Add your remote GitHub repo URL (create an empty repo on github.com first)
git remote add origin https://github.com/YOUR_USERNAME/resume-analyzer.git

# Push to GitHub
git push -u origin main
```

---

## 🔮 Future Improvements

- [ ] AI-assisted bullet point rewriter using local Ollama or HuggingFace models.
- [ ] Multi-language resume parsing (Spanish, German, French).
- [ ] Direct export to LinkedIn profile summary format.
- [ ] LaTeX resume template generator based on extracted entities.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
