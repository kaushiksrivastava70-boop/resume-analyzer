"""Recruiter Mode Batch Processing & Analytics Engine.

Provides in-memory ZIP parsing, candidate metadata extraction,
deterministic TF-IDF cosine similarity scoring, skill gap aggregation,
interactive leaderboard ranking, Plotly analytics, and CSV/Excel exports.
"""

from __future__ import annotations

import io
import math
import re
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analyzer import (
    ContactInfo,
    ResumeExtractionError,
    extract_contact_info,
    extract_skills,
    extract_text,
)
from chatgpt_evaluator import (
    ChatGptEvaluationResult,
    classify_jd_requirements,
    evaluate_resume_chatgpt,
    generate_chatgpt_prompt,
    match_skills_semantically,
)


@dataclass
class CandidateResult:
    """Stores parsed evaluation results for an individual candidate."""
    filename: str
    name: str
    email: Optional[str]
    phone: Optional[str]
    linkedin: Optional[str]
    github: Optional[str]
    years_of_experience: Optional[float]
    all_skills: List[str]
    matched_skills: List[str]
    missing_skills: List[str]
    skill_match_percentage: float
    tfidf_similarity: float
    composite_score: float
    key_bullet_points: List[str]
    raw_text: str
    status: str = "Success"
    error_message: Optional[str] = None
    chatgpt_evaluation: Optional[ChatGptEvaluationResult] = None


@dataclass
class BatchAnalysisResult:
    """Stores aggregated results and analytics across all processed resumes."""
    candidates: List[CandidateResult]
    total_scanned: int
    shortlisted_count: int
    average_score: float
    top_missing_skills: List[Tuple[str, int]]
    jd_skills: List[str]
    errors: List[Dict[str, str]] = field(default_factory=list)


# =========================================================================
# 1. CANDIDATE METADATA & TEXT EXTRACTION HELPERS
# =========================================================================

def clean_filename_to_name(filename: str) -> str:
    """Derive a plausible candidate name from the file name as a fallback."""
    # Remove extension
    base = re.sub(r"\.[^.]+$", "", filename)
    # Replace separators with spaces first so word boundaries work on words separated by underscores/hyphens
    base = re.sub(r"[_\-\.]+", " ", base)
    # Remove common boilerplate keywords
    base = re.sub(
        r"(?i)\b(resume|cv|curriculum|vitae|final|latest|updated|v\d+|\d{4}|copy|draft)\b",
        "",
        base,
    )
    # Clean up whitespace
    clean = re.sub(r"\s+", " ", base).strip()
    words = [w.capitalize() for w in clean.split() if w.isalpha() and len(w) > 1]
    if len(words) >= 2:
        return " ".join(words[:3])
    elif len(words) == 1:
        return words[0]
    return "Candidate"


def extract_candidate_name(text: str, filename: str) -> str:
    """Extract candidate name using top-of-resume heuristics with filename fallback."""
    if not text or not text.strip():
        return clean_filename_to_name(filename)

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if not lines:
        return clean_filename_to_name(filename)

    # Common non-name headers or tokens to disqualify
    blacklist = {
        "resume", "curriculum", "vitae", "summary", "profile", "contact",
        "experience", "work", "education", "skills", "projects", "objective",
        "professional", "technical", "page", "phone", "email", "address",
        "portfolio", "github", "linkedin", "overview"
    }

    # Inspect the first 8 non-empty lines
    for line in lines[:8]:
        # Disqualify if contains email, phone, links, or digits
        if "@" in line or "http" in line or "www." in line or "+" in line or re.search(r"\d", line):
            continue

        clean_line = re.sub(r"[,|\•\-\–—\(\)]", " ", line)
        words = clean_line.split()

        # Plausible names are usually 2 to 4 words
        if 2 <= len(words) <= 4:
            # Check if all words are alphabetic
            if all(re.match(r"^[a-zA-Z\.\']+$", w) for w in words):
                words_lower = [w.lower().rstrip(".") for w in words]
                # Ensure no word is a blacklisted header
                if not any(w in blacklist for w in words_lower):
                    # Check that each word starts with a capital or the whole line is uppercase
                    is_title_or_upper = all(w[0].isupper() for w in words) or line.isupper()
                    if is_title_or_upper and 3 <= len(clean_line) <= 40:
                        return " ".join(w.capitalize() for w in words)

    # Fallback to cleaned filename
    return clean_filename_to_name(filename)


def extract_years_of_experience(text: str) -> Optional[float]:
    """Extract total years of experience using regex patterns and date span heuristics."""
    if not text:
        return None

    word_to_num = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20
    }

    found_years: List[float] = []

    # 1. Stated experience pattern: "5+ years of experience", "over 3.5 yrs", "7 years in software"
    exp_pattern = re.compile(
        r"(?i)\b(?:over|around|more than|approximately)?\s*"
        r"(\d+(?:\.\d+)?|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty)\b)\+?\s*"
        r"(?:years?|yrs?)(?:\s+of)?(?:\s+(?:hands-on|professional|work|industry|software|engineering|relevant|full-time|proven))?\s*experience\b"
    )

    for match in exp_pattern.finditer(text):
        val_str = match.group(1).lower()
        if val_str in word_to_num:
            val = float(word_to_num[val_str])
        else:
            try:
                val = float(val_str)
            except ValueError:
                continue
        if 0.5 <= val <= 40.0:
            found_years.append(val)

    if found_years:
        return max(found_years)

    # 2. Year-range heuristic: e.g. "2018 - 2024", "2019 - Present"
    year_ranges = re.findall(
        r"\b(20\d\d|19\d\d)\s*(?:[-–—]|to)\s*(20\d\d|present|current)\b",
        text,
        re.IGNORECASE,
    )
    if year_ranges:
        current_year = 2026  # Anchored to current environment year
        total_span = 0.0
        for start_str, end_str in year_ranges:
            try:
                start = int(start_str)
                end = current_year if end_str.lower() in ("present", "current") else int(end_str)
                span = end - start
                if 0 <= span <= 25:
                    total_span = max(total_span, float(span))
            except Exception:
                continue
        if total_span > 0:
            return round(total_span, 1)

    return None


def extract_bullet_points(text: str, max_bullets: int = 5) -> List[str]:
    """Extract notable achievement bullet points from resume text."""
    if not text:
        return []

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    bullet_candidates = []

    for line in lines:
        cleaned = re.sub(r"^[\*\•\-\–—▪\d\.]+\s*", "", line).strip()
        # Ensure sufficient length and descriptive content
        if 35 <= len(cleaned) <= 220:
            # Check for strong impact indicators or verbs
            if re.search(r"\b(\d+%|\$\d+|built|designed|developed|architected|led|managed|scaled|optimized|increased|reduced)\b", cleaned, re.IGNORECASE):
                bullet_candidates.append(cleaned)
                if len(bullet_candidates) >= max_bullets:
                    break

    # If no impact bullets found, take the first descriptive lines
    if not bullet_candidates:
        for line in lines[3:]:
            cleaned = re.sub(r"^[\*\•\-\–—▪\d\.]+\s*", "", line).strip()
            if 40 <= len(cleaned) <= 180 and not any(h in cleaned.lower() for h in ["education", "skills", "experience"]):
                bullet_candidates.append(cleaned)
                if len(bullet_candidates) >= max_bullets:
                    break

    return bullet_candidates


# =========================================================================
# 2. PURE-PYTHON DETERMINISTIC TF-IDF & COSINE SIMILARITY ENGINE
# =========================================================================

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves", "role", "looking", "candidate", "responsibilities",
    "opportunity", "requirements", "qualifications", "company", "team", "work"
}


def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric words excluding stopwords."""
    words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9_\-\.]{1,}\b", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 2]


class SimpleTfidfCosineMatcher:
    """Computes TF-IDF vector representations and Cosine Similarities."""

    def __init__(self, documents: List[str]):
        """Build vocabulary and inverse document frequencies across documents."""
        self.doc_tokens = [tokenize(doc) for doc in documents]
        self.total_docs = len(documents)

        # Compute document frequency df(t) for every term
        df_counter: Counter[str] = Counter()
        for tokens in self.doc_tokens:
            unique_terms = set(tokens)
            df_counter.update(unique_terms)

        self.vocabulary: Dict[str, int] = {
            term: idx for idx, (term, count) in enumerate(df_counter.items())
        }

        # IDF with smooth: idf(t) = ln((1 + N) / (1 + df(t))) + 1.0
        self.idf: Dict[str, float] = {}
        for term, df in df_counter.items():
            self.idf[term] = math.log((1.0 + self.total_docs) / (1.0 + df)) + 1.0

    def transform(self, text: str) -> Dict[str, float]:
        """Convert a text string to an L2-normalized TF-IDF vector."""
        tokens = tokenize(text)
        if not tokens:
            return {}

        tf_counter = Counter(tokens)
        total_tokens = len(tokens)

        # Vector: {term: tf * idf}
        vector: Dict[str, float] = {}
        sum_sq = 0.0
        for term, count in tf_counter.items():
            if term in self.idf:
                tf = count / total_tokens
                weight = tf * self.idf[term]
                vector[term] = weight
                sum_sq += weight * weight

        # L2 Normalize vector
        if sum_sq > 0.0:
            norm = math.sqrt(sum_sq)
            for term in vector:
                vector[term] /= norm

        return vector

    @staticmethod
    def cosine_similarity(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
        """Compute cosine similarity between two normalized TF-IDF vectors (0.0 to 1.0)."""
        if not vec_a or not vec_b:
            return 0.0

        # Dot product of normalized vectors
        dot_product = 0.0
        # Iterate over smaller dict
        small_vec, large_vec = (vec_a, vec_b) if len(vec_a) < len(vec_b) else (vec_b, vec_a)
        for term, val in small_vec.items():
            if term in large_vec:
                dot_product += val * large_vec[term]

        return max(0.0, min(1.0, dot_product))


def compute_tfidf_similarities(jd_text: str, resume_texts: List[str]) -> List[float]:
    """Calculate TF-IDF cosine similarity scores (0-100%) for each resume against JD."""
    if not jd_text.strip() or not resume_texts:
        return [0.0] * len(resume_texts)

    # All documents include the JD and all resumes
    corpus = [jd_text] + resume_texts
    engine = SimpleTfidfCosineMatcher(corpus)

    jd_vector = engine.transform(jd_text)
    scores = []
    for r_text in resume_texts:
        r_vector = engine.transform(r_text)
        sim = engine.cosine_similarity(jd_vector, r_vector)
        scores.append(round(sim * 100.0, 1))

    return scores


# =========================================================================
# 3. BATCH PROCESSING PIPELINE (IN-MEMORY ZIP)
# =========================================================================

def process_resume_zip(
    zip_bytes: bytes,
    jd_text: str,
    threshold: float = 60.0,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    openai_api_key: Optional[str] = None,
    openai_model: str = "gpt-4o-mini",
) -> BatchAnalysisResult:
    """Unpack in-memory ZIP and run deterministic scoring & parsing pipeline."""
    if not zip_bytes:
        raise ValueError("Provided ZIP file is empty (0 bytes).")

    try:
        zip_buffer = io.BytesIO(zip_bytes)
        archive = zipfile.ZipFile(zip_buffer)
    except zipfile.BadZipFile:
        raise ValueError("The uploaded file is not a valid ZIP archive.")

    # Extract target JD skills
    _, jd_skills = extract_skills(jd_text)
    jd_skills_lower = {s.lower() for s in jd_skills}

    # Filter valid resume file entries
    valid_entries: List[zipfile.ZipInfo] = []
    for info in archive.infolist():
        # Skip directories and Mac OS metadata files
        if info.is_dir() or info.filename.startswith("__MACOSX/") or "/." in info.filename or info.filename.startswith("."):
            continue
        lower_name = info.filename.lower()
        if lower_name.endswith((".pdf", ".docx", ".txt")):
            valid_entries.append(info)

    if not valid_entries:
        raise ValueError("No supported resume files (.pdf, .docx, or .txt) were found inside the ZIP.")

    candidates: List[CandidateResult] = []
    errors: List[Dict[str, str]] = []
    extracted_texts: List[str] = []
    valid_entry_indices: List[int] = []

    total_files = len(valid_entries)

    # 1. Text Extraction Phase
    for idx, entry in enumerate(valid_entries):
        clean_name = entry.filename.split("/")[-1]
        if progress_callback:
            progress_callback((idx / (total_files * 2)), f"Extracting {clean_name}...")

        try:
            file_data = archive.read(entry)
            text = extract_text(file_data, clean_name)
            extracted_texts.append(text)
            valid_entry_indices.append(idx)
        except ResumeExtractionError as e:
            errors.append({"file": clean_name, "error": str(e)})
            candidates.append(
                CandidateResult(
                    filename=clean_name,
                    name=clean_filename_to_name(clean_name),
                    email=None,
                    phone=None,
                    linkedin=None,
                    github=None,
                    years_of_experience=None,
                    all_skills=[],
                    matched_skills=[],
                    missing_skills=jd_skills,
                    skill_match_percentage=0.0,
                    tfidf_similarity=0.0,
                    composite_score=0.0,
                    key_bullet_points=[],
                    raw_text="",
                    status="Error",
                    error_message=str(e),
                )
            )
        except Exception as e:
            err_msg = f"Unexpected read failure: {str(e)}"
            errors.append({"file": clean_name, "error": err_msg})
            candidates.append(
                CandidateResult(
                    filename=clean_name,
                    name=clean_filename_to_name(clean_name),
                    email=None,
                    phone=None,
                    linkedin=None,
                    github=None,
                    years_of_experience=None,
                    all_skills=[],
                    matched_skills=[],
                    missing_skills=jd_skills,
                    skill_match_percentage=0.0,
                    tfidf_similarity=0.0,
                    composite_score=0.0,
                    key_bullet_points=[],
                    raw_text="",
                    status="Error",
                    error_message=err_msg,
                )
            )

    # 2. Batch TF-IDF Calculation
    tfidf_scores: List[float] = []
    if extracted_texts and jd_text.strip():
        tfidf_scores = compute_tfidf_similarities(jd_text, extracted_texts)
    else:
        tfidf_scores = [0.0] * len(extracted_texts)

    # 3. Candidate Metadata & ChatGPT-Grade Semantic Scoring Phase
    for local_idx, global_idx in enumerate(valid_entry_indices):
        entry = valid_entries[global_idx]
        clean_name = entry.filename.split("/")[-1]
        text = extracted_texts[local_idx]
        tfidf_sim = tfidf_scores[local_idx]

        if progress_callback:
            progress_callback(
                0.5 + (local_idx / (len(valid_entry_indices) * 2)),
                f"Running semantic analysis for {clean_name}...",
            )

        name = extract_candidate_name(text, clean_name)
        contact = extract_contact_info(text)
        years_exp = extract_years_of_experience(text)
        _, candidate_skills = extract_skills(text)
        bullets = extract_bullet_points(text, max_bullets=4)

        # ChatGPT-aligned semantic evaluation
        chatgpt_eval = evaluate_resume_chatgpt(
            resume_text=text,
            jd_text=jd_text,
            candidate_name=name,
            api_key=openai_api_key,
            model=openai_model,
        )

        matched_skills = chatgpt_eval.mandatory_matched + chatgpt_eval.preferred_matched
        missing_skills = chatgpt_eval.mandatory_missing + chatgpt_eval.preferred_missing

        if jd_skills:
            skill_match_pct = round((len(matched_skills) / len(jd_skills)) * 100.0, 1)
        else:
            skill_match_pct = 0.0

        # Composite score: incorporates ChatGPT semantic score + TF-IDF semantic alignment
        composite = round((0.75 * chatgpt_eval.chatgpt_score) + (0.25 * tfidf_sim), 1)

        candidates.append(
            CandidateResult(
                filename=clean_name,
                name=name,
                email=contact.email,
                phone=contact.phone,
                linkedin=contact.linkedin,
                github=contact.github,
                years_of_experience=years_exp,
                all_skills=candidate_skills,
                matched_skills=matched_skills,
                missing_skills=missing_skills,
                skill_match_percentage=skill_match_pct,
                tfidf_similarity=tfidf_sim,
                composite_score=composite,
                key_bullet_points=bullets,
                raw_text=text,
                status="Success",
                chatgpt_evaluation=chatgpt_eval,
            )
        )

    # Sort candidates by Composite Score descending
    candidates.sort(key=lambda c: (c.status == "Success", c.composite_score), reverse=True)

    # Aggregate Analytics
    successful_candidates = [c for c in candidates if c.status == "Success"]
    total_scanned = len(candidates)
    shortlisted_count = sum(1 for c in successful_candidates if c.composite_score >= threshold)
    avg_score = (
        round(sum(c.composite_score for c in successful_candidates) / len(successful_candidates), 1)
        if successful_candidates
        else 0.0
    )

    # Compute top missing skills across all applicants
    missing_skill_counter: Counter[str] = Counter()
    for c in successful_candidates:
        missing_skill_counter.update(c.missing_skills)

    top_missing = missing_skill_counter.most_common(5)

    if progress_callback:
        progress_callback(1.0, "Batch analysis complete!")

    return BatchAnalysisResult(
        candidates=candidates,
        total_scanned=total_scanned,
        shortlisted_count=shortlisted_count,
        average_score=avg_score,
        top_missing_skills=top_missing,
        jd_skills=jd_skills,
        errors=errors,
    )


# =========================================================================
# 4. EXPORT ENGINE (CSV & EXCEL)
# =========================================================================

def candidates_to_dataframe(candidates: List[CandidateResult], threshold: float = 60.0) -> pd.DataFrame:
    """Convert candidates list into a structured pandas DataFrame."""
    rows = []
    for idx, c in enumerate(candidates, start=1):
        if c.status == "Success":
            status_label = "Shortlisted ✅" if c.composite_score >= threshold else "Under Review ⚠️"
        else:
            status_label = "Parse Error ❌"

        ai_verdict = c.chatgpt_evaluation.verdict if c.chatgpt_evaluation else "N/A"
        rows.append({
            "Rank": idx,
            "Status": status_label,
            "Candidate Name": c.name,
            "Match Score (%)": c.composite_score,
            "AI Verdict": ai_verdict,
            "Skill Match (%)": c.skill_match_percentage,
            "TF-IDF Sim (%)": c.tfidf_similarity,
            "Years Exp": f"{c.years_of_experience:.1f} yrs" if c.years_of_experience is not None else "N/A",
            "Matched Skills": ", ".join(c.matched_skills) if c.matched_skills else "None",
            "Missing Skills": ", ".join(c.missing_skills) if c.missing_skills else "None",
            "Email": c.email or "N/A",
            "Phone": c.phone or "N/A",
            "Filename": c.filename,
        })
    return pd.DataFrame(rows)


def export_to_csv(df: pd.DataFrame) -> bytes:
    """Generate CSV bytes from DataFrame."""
    return df.to_csv(index=False).encode("utf-8")


def export_to_excel(df: pd.DataFrame) -> bytes:
    """Generate styled Excel workbook bytes from DataFrame using openpyxl."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Candidate Rankings")
        workbook = writer.book
        worksheet = writer.sheets["Candidate Rankings"]

        # Basic styling for headers
        from openpyxl.styles import Alignment, Font, PatternFill

        header_fill = PatternFill(start_color="4F46E5", end_color="4F46E5", fill_type="solid")
        header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")

        for cell in worksheet[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center")

        # Auto-adjust column widths
        for col in worksheet.columns:
            max_len = max(len(str(cell.value or "")) for cell in col)
            col_letter = col[0].column_letter
            worksheet.column_dimensions[col_letter].width = max(max_len + 3, 12)

    return output.getvalue()


# =========================================================================
# 5. DEMO BATCH GENERATOR FOR 1-CLICK TESTING
# =========================================================================

def create_demo_resume_zip() -> bytes:
    """Generate an in-memory ZIP archive containing 5 diverse candidate resumes."""
    resumes = {
        "Alex_Rivera_Senior_FullStack.txt": """ALEX RIVERA
alex.rivera@email.com | +1 (555) 492-8172 | San Francisco, CA
linkedin.com/in/alexrivera-eng | github.com/alexrivera

PROFESSIONAL SUMMARY
Senior Full-Stack Engineer with 6+ years of experience building high-scale cloud platforms.
Specialized in Python, React, FastAPI, AWS microservices, and high-performance databases.

WORK EXPERIENCE
Lead Software Engineer | Datacore Cloud (2021 - Present)
- Architected enterprise cloud services using Python, FastAPI, React, and PostgreSQL serving 15M+ requests daily.
- Deployed Docker containers to Kubernetes on AWS ECS, reducing deployment latency by 55%.
- Implemented Redis caching layers and GraphQL endpoints, improving API latency by 40%.

Senior Backend Developer | ScaleWorks (2018 - 2021)
- Developed REST microservices with Python, Django, and PostgreSQL.
- Automated CI/CD pipelines with GitHub Actions, Terraform, and Docker.

TECHNICAL SKILLS
Languages: Python, TypeScript, JavaScript, SQL, Golang
Frameworks: React, Next.js, FastAPI, Node.js, GraphQL
Cloud & DevOps: AWS, Docker, Kubernetes, Terraform, Redis, PostgreSQL, Git

EDUCATION
B.S. in Computer Science | UC Berkeley
""",
        "Priya_Sharma_Cloud_DevOps.txt": """PRIYA SHARMA
priya.sharma@cloudtech.io | +1 (555) 901-2384 | Seattle, WA
linkedin.com/in/priyasharma-devops | github.com/priyadev

PROFESSIONAL SUMMARY
DevOps & Cloud Infrastructure Architect with 5+ years of experience leading automated AWS cloud transformations.

WORK EXPERIENCE
Senior Cloud Architect | CloudNative Labs (2020 - Present)
- Engineered scalable infrastructure on AWS using Terraform and Kubernetes.
- Optimized PostgreSQL database clusters and distributed Redis caches for high availability.
- Maintained Python monitoring scripts and automated CI/CD deployment workflows with Docker.

DevOps Engineer | FinTech Innovations (2019 - 2020)
- Configured Docker containers and microservices monitoring with Prometheus.

SKILLS
Cloud: AWS, Docker, Kubernetes, Terraform, Linux
Databases: PostgreSQL, Redis, MongoDB
Languages: Python, Bash, Go
Frameworks: FastAPI, Flask

EDUCATION
B.Tech in Computer Engineering | University of Washington
""",
        "Jordan_Lee_Frontend_React.txt": """JORDAN LEE
jordan.lee@devmail.com | +1 (555) 234-8901 | New York, NY
linkedin.com/in/jordanlee-frontend | github.com/jordanlee

PROFESSIONAL SUMMARY
Frontend Web Specialist with 4+ years designing responsive modern UI applications using React, Next.js, and TypeScript.

WORK EXPERIENCE
Senior Frontend Developer | UI Innovations (2021 - Present)
- Created modular design system components using React, TypeScript, Next.js, and Tailwind CSS.
- Integrated GraphQL and RESTful APIs, reducing page load time by 35%.
- Partnered with product and UX design teams to elevate user accessibility.

Frontend Engineer | WebSphere Agency (2020 - 2021)
- Built interactive client web portals using JavaScript, React, and CSS3.

SKILLS
Frontend: React, Next.js, TypeScript, JavaScript, HTML5, CSS3, Redux
Tools: Git, Webpack, Figma
Backend Exposure: Node.js, Express, REST APIs

EDUCATION
B.A. in Interactive Digital Media | NYU
""",
        "Sam_Taylor_Junior_Engineer.txt": """SAM TAYLOR
sam.taylor@techhub.org | +1 (555) 678-1234 | Austin, TX
linkedin.com/in/samtaylor-junior | github.com/samtaylor

PROFESSIONAL SUMMARY
Enthusiastic Junior Software Engineer with 1.5 years experience in Python scripting and web fundamentals.

EXPERIENCE
Junior Software Developer | CodeBase Solutions (2023 - Present)
- Assisted backend team in writing unit tests and bug fixes in Python.
- Built internal administrative tools with Flask and SQLite.
- Participated in weekly code reviews and agile sprint planning.

TECHNICAL SKILLS
Languages: Python, JavaScript, HTML, CSS, SQL
Frameworks: Flask, Django
Tools: Git, Linux, SQLite

EDUCATION
B.S. in Information Systems | UT Austin
""",
        "Elena_Rostova_Data_ML_Engineer.txt": """ELENA ROSTOVA
elena.rostova@datascience.net | +1 (555) 345-6789 | Boston, MA
linkedin.com/in/elenarostova-data | github.com/erostova

PROFESSIONAL SUMMARY
Machine Learning & Data Engineer with 4+ years developing predictive models and data analytics pipelines.

WORK EXPERIENCE
Data & ML Engineer | Apex Intelligence (2021 - Present)
- Built automated data processing pipelines in Python, Pandas, and PyTorch.
- Queried large datasets with PostgreSQL and Snowflake, training classification algorithms.
- Containerized model inference microservices with Docker and FastAPI on AWS.

Data Analyst | DataMetric Co. (2020 - 2021)
- Built automated BI reporting dashboards and conducted statistical analyses.

SKILLS
ML & Data: Python, PyTorch, Scikit-Learn, Pandas, NumPy, SQL
Databases & Cloud: PostgreSQL, AWS, Docker, Git
Web & APIs: FastAPI, Flask

EDUCATION
M.S. in Data Science | MIT
""",
    }

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for filename, content in resumes.items():
            zf.writestr(filename, content)

    return buf.getvalue()


# =========================================================================
# 6. STREAMLIT UI: RENDER RECRUITER MODE
# =========================================================================

def render_recruiter_mode():
    """Render the Recruiter Mode batch screening and leaderboard interface."""
    st.markdown(
        """
        <div class="hero-container" style="background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #334155 100%);">
            <div class="hero-badge" style="background: rgba(99, 102, 241, 0.3); border: 1px solid #6366F1;">
                🏢 Recruiter & Hiring Manager Hub
            </div>
            <div class="hero-title">Batch Resume Screening & Leaderboard</div>
            <p class="hero-subtitle">
                Upload a ZIP archive of candidate resumes (PDF, DOCX, TXT) alongside your target Job Description.
                Instantly score, rank, analyze skill gaps, and export ranked shortlists.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- INPUT SECTION: JD & ZIP UPLOAD ---
    st.markdown("### 📋 1. Job Description & Resume Archive")

    default_jd = """Senior Full-Stack Engineer

About the Role:
We are looking for a Senior Full-Stack Engineer with 4+ years of hands-on experience building high-scale cloud web systems.

Requirements & Qualifications:
- 4+ years software engineering experience with Python, TypeScript, and React.
- Strong proficiency in modern API frameworks (FastAPI, Next.js, GraphQL).
- Production expertise with AWS, Docker, Kubernetes, and Terraform.
- Solid experience with relational databases (PostgreSQL) and in-memory caches (Redis).
- Familiarity with Golang, Kafka, and PyTorch is a strong plus.
- Proven track record of optimizing system performance and mentoring teammates."""

    col_jd, col_zip = st.columns([1, 1], gap="large")

    with col_jd:
        st.markdown("**Target Job Description (JD)**")
        jd_input = st.text_area(
            "Paste Job Description:",
            value=st.session_state.get("recruiter_jd", default_jd),
            height=210,
            placeholder="Paste role requirements, duties, and target qualifications...",
            key="recruiter_jd_input",
        )

    with col_zip:
        st.markdown("**Upload Candidate Resumes (.ZIP)**")
        uploaded_zip = st.file_uploader(
            "Upload ZIP containing PDF/DOCX resumes",
            type=["zip"],
            help="All files inside the ZIP will be processed directly in-memory without saving to server disk.",
            key="recruiter_zip_uploader",
        )

        col_demo_btn, col_demo_clear = st.columns([2, 1])
        with col_demo_btn:
            if st.button("✨ Load Demo Candidate Batch (5 Resumes)", type="secondary", use_container_width=True):
                st.session_state["recruiter_use_demo"] = True
                st.session_state["demo_zip_bytes"] = create_demo_resume_zip()
                st.rerun()

        with col_demo_clear:
            if st.session_state.get("recruiter_use_demo", False):
                if st.button("🔄 Reset Demo", use_container_width=True):
                    st.session_state["recruiter_use_demo"] = False
                    st.session_state.pop("demo_zip_bytes", None)
                    st.session_state.pop("recruiter_results", None)
                    st.rerun()

    # --- THRESHOLD & AI SETTINGS ---
    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("### ⚙️ 2. Screening Parameters & AI Evaluator")
    col_thresh, col_ai_opts = st.columns([1, 2], gap="large")
    with col_thresh:
        pass_threshold = st.slider(
            "Minimum Shortlist Passing Score (%)",
            min_value=30.0,
            max_value=95.0,
            value=60.0,
            step=5.0,
            help="Candidates scoring at or above this threshold will be flagged as 'Shortlisted ✅'.",
        )
    with col_ai_opts:
        with st.expander("🤖 ChatGPT / OpenAI Integration Settings (Optional)", expanded=False):
            st.caption("By default, the built-in **ChatGPT Semantic Emulator** runs 100% offline at zero cost. You can also provide an OpenAI API key for live GPT-4o evaluations.")
            col_k1, col_k2 = st.columns([2, 1])
            with col_k1:
                openai_api_key_input = st.text_input(
                    "OpenAI API Key (Optional):",
                    value=st.session_state.get("openai_api_key", ""),
                    type="password",
                    placeholder="sk-...",
                    key="recruiter_api_key_input",
                )
                if openai_api_key_input.strip():
                    st.session_state["openai_api_key"] = openai_api_key_input.strip()
                elif "openai_api_key" in st.session_state and not openai_api_key_input:
                    st.session_state.pop("openai_api_key", None)
            with col_k2:
                openai_model_choice = st.selectbox(
                    "Model:",
                    options=["gpt-4o-mini", "gpt-4o"],
                    index=0,
                    key="recruiter_model_choice",
                )
            if st.session_state.get("openai_api_key"):
                st.success(f"🟢 Live OpenAI API Connected ({openai_model_choice})")
            else:
                st.info("⚡ Built-in ChatGPT Semantic Emulator Active (100% Offline & Free)")

    # Determine data source: uploaded zip or demo zip
    zip_bytes_to_process = None
    if uploaded_zip is not None:
        zip_bytes_to_process = uploaded_zip.read()
    elif st.session_state.get("recruiter_use_demo", False):
        zip_bytes_to_process = st.session_state.get("demo_zip_bytes", None)
        st.info("ℹ️ Using **Demo Candidate Batch** (5 realistic engineering resumes: Full-Stack, Cloud/DevOps, Frontend, Junior, and ML/Data).")

    if not zip_bytes_to_process:
        st.info("👆 Upload a `.zip` archive containing resumes or click **'Load Demo Candidate Batch'** above to begin.")
        return

    if not jd_input.strip():
        st.warning("⚠️ Please provide a target Job Description to compare candidate resumes against.")
        return

    # --- PROCESS BATCH ---
    run_batch = False
    effective_api_key = st.session_state.get("openai_api_key", "")
    current_key = f"{len(zip_bytes_to_process)}_{len(jd_input)}_{effective_api_key}_{openai_model_choice}"
    if st.session_state.get("last_processed_key") != current_key:
        run_batch = True

    if run_batch:
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def update_progress(ratio: float, msg: str):
            progress_bar.progress(min(1.0, max(0.0, ratio)))
            status_text.text(msg)

        try:
            with st.spinner("Processing batch resumes with ChatGPT semantic analysis..."):
                results = process_resume_zip(
                    zip_bytes=zip_bytes_to_process,
                    jd_text=jd_input,
                    threshold=pass_threshold,
                    progress_callback=update_progress,
                    openai_api_key=effective_api_key if effective_api_key else None,
                    openai_model=openai_model_choice,
                )
                st.session_state["recruiter_results"] = results
                st.session_state["last_processed_key"] = current_key
            progress_bar.empty()
            status_text.empty()
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Failed to process resume batch: {str(e)}")
            return
    else:
        results: BatchAnalysisResult = st.session_state["recruiter_results"]
        # Update shortlisting count dynamically if threshold changed
        results.shortlisted_count = sum(
            1 for c in results.candidates if c.status == "Success" and c.composite_score >= pass_threshold
        )

    st.markdown("---")

    # =========================================================================
    # SUMMARY KPI METRIC CARDS
    # =========================================================================
    st.markdown("### 📊 3. Screening Analytics & Insights")

    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    with col_kpi1:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-value">{results.total_scanned}</div>
                <div class="stat-label">Total Resumes Scanned</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_kpi2:
        shortlist_pct = round((results.shortlisted_count / max(1, results.total_scanned)) * 100, 1)
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-value" style="color: #10B981;">{results.shortlisted_count} <span style="font-size:1rem; color:#64748B;">({shortlist_pct}%)</span></div>
                <div class="stat-label">Shortlisted (≥ {pass_threshold:.0f}%)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_kpi3:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-value" style="color: #4F46E5;">{results.average_score}%</div>
                <div class="stat-label">Average Match Score</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_kpi4:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-value">{len(results.jd_skills)}</div>
                <div class="stat-label">Target JD Skills Required</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br/>", unsafe_allow_html=True)

    # --- PLOTLY ANALYTICS CHARTS ---
    col_chart_skills, col_chart_dist = st.columns([1, 1], gap="medium")

    with col_chart_skills:
        st.markdown("#### 📉 Top Missing Skills Across All Applicants")
        if results.top_missing_skills:
            skill_names = [item[0] for item in reversed(results.top_missing_skills)]
            skill_counts = [item[1] for item in reversed(results.top_missing_skills)]

            fig_missing = go.Figure(
                go.Bar(
                    x=skill_counts,
                    y=skill_names,
                    orientation="h",
                    marker=dict(
                        color="#EF4444",
                        opacity=0.85,
                        line=dict(color="#B91C1C", width=1.5),
                    ),
                    text=[f"{cnt} candidate{'s' if cnt > 1 else ''}" for cnt in skill_counts],
                    textposition="auto",
                )
            )
            fig_missing.update_layout(
                xaxis_title="Number of Candidates Missing Skill",
                yaxis_title="",
                height=280,
                margin=dict(l=10, r=20, t=10, b=30),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter"),
                xaxis=dict(gridcolor="#E2E8F0", dtick=1),
            )
            st.plotly_chart(fig_missing, use_container_width=True)
            st.caption("Identifies widespread skill gaps in your applicant pool to calibrate job requirements.")
        else:
            st.success("🎉 All candidates possess 100% of the target skills!")

    with col_chart_dist:
        st.markdown("#### 🎯 Applicant Score Distribution")
        scores = [c.composite_score for c in results.candidates if c.status == "Success"]
        names = [c.name for c in results.candidates if c.status == "Success"]

        if scores:
            fig_dist = px.bar(
                x=names,
                y=scores,
                labels={"x": "Candidate", "y": "Match Score (%)"},
                color=scores,
                color_continuous_scale=["#EF4444", "#F59E0B", "#10B981"],
                range_color=[0, 100],
            )
            fig_dist.add_hline(
                y=pass_threshold,
                line_dash="dash",
                line_color="#1E293B",
                annotation_text=f"Threshold ({pass_threshold:.0f}%)",
                annotation_position="bottom right",
            )
            fig_dist.update_layout(
                height=280,
                margin=dict(l=10, r=20, t=10, b=30),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter"),
                yaxis=dict(range=[0, 105], gridcolor="#E2E8F0"),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_dist, use_container_width=True)
            st.caption("Visual ranking of applicant scores relative to your passing threshold.")
        else:
            st.info("No successful resume scores available.")

    # =========================================================================
    # LEADERBOARD DATAFRAME & EXPORT
    # =========================================================================
    st.markdown("---")
    st.markdown("### 🏆 4. Candidate Screening Leaderboard")

    df = candidates_to_dataframe(results.candidates, threshold=pass_threshold)

    # Filter controls
    col_filter1, col_filter2, col_export_csv, col_export_xlsx = st.columns([2, 2, 1.5, 1.5], gap="small")

    with col_filter1:
        status_filter = st.selectbox(
            "Filter by Status:",
            options=["All Candidates", "Shortlisted Only ✅", "Under Review Only ⚠️"],
            key="recruiter_status_filter",
        )

    with col_filter2:
        search_query = st.text_input(
            "Search Candidates / Skills:",
            placeholder="Search by name, skill, or keyword...",
            key="recruiter_search_query",
        )

    # Apply filters
    filtered_df = df.copy()
    if status_filter == "Shortlisted Only ✅":
        filtered_df = filtered_df[filtered_df["Status"].str.contains("Shortlisted")]
    elif status_filter == "Under Review Only ⚠️":
        filtered_df = filtered_df[filtered_df["Status"].str.contains("Under Review")]

    if search_query.strip():
        q = search_query.strip().lower()
        filtered_df = filtered_df[
            filtered_df["Candidate Name"].str.lower().str.contains(q)
            | filtered_df["Matched Skills"].str.lower().str.contains(q)
            | filtered_df["Missing Skills"].str.lower().str.contains(q)
            | filtered_df["Filename"].str.lower().str.contains(q)
        ]

    # Export Buttons
    with col_export_csv:
        csv_bytes = export_to_csv(filtered_df)
        st.download_button(
            label="📥 Export CSV",
            data=csv_bytes,
            file_name="recruiter_screening_leaderboard.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col_export_xlsx:
        xlsx_bytes = export_to_excel(filtered_df)
        st.download_button(
            label="📊 Export Excel",
            data=xlsx_bytes,
            file_name="recruiter_screening_leaderboard.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    # Interactive Leaderboard Dataframe
    st.dataframe(
        filtered_df,
        use_container_width=True,
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", width="small"),
            "Status": st.column_config.TextColumn("Status", width="medium"),
            "Candidate Name": st.column_config.TextColumn("Candidate Name", width="medium"),
            "Match Score (%)": st.column_config.ProgressColumn(
                "Match Score (%)",
                format="%.1f%%",
                min_value=0,
                max_value=100,
                width="medium",
            ),
            "Skill Match (%)": st.column_config.NumberColumn("Skill Match", format="%.1f%%", width="small"),
            "TF-IDF Sim (%)": st.column_config.NumberColumn("TF-IDF Sim", format="%.1f%%", width="small"),
            "Years Exp": st.column_config.TextColumn("Experience", width="small"),
            "Matched Skills": st.column_config.TextColumn("Matched Skills", width="large"),
            "Missing Skills": st.column_config.TextColumn("Missing Skills", width="large"),
            "Email": st.column_config.TextColumn("Email", width="medium"),
            "Phone": st.column_config.TextColumn("Phone", width="medium"),
            "Filename": st.column_config.TextColumn("Filename", width="medium"),
        },
        hide_index=True,
    )

    if results.errors:
        with st.expander(f"⚠️ Files with Extraction Errors ({len(results.errors)})", expanded=False):
            for err in results.errors:
                st.error(f"**{err['file']}**: {err['error']}")

    # =========================================================================
    # CANDIDATE DETAIL DRILLDOWN
    # =========================================================================
    st.markdown("---")
    st.markdown("### 🔍 5. Candidate Dossier Drilldown")

    successful_candidates = [c for c in results.candidates if c.status == "Success"]
    if not successful_candidates:
        st.info("No candidates available for detailed inspection.")
        return

    candidate_options = {f"{c.name} ({c.composite_score:.1f}% Match - {c.filename})": c for c in successful_candidates}
    selected_label = st.selectbox(
        "Select candidate to inspect detailed profile:",
        options=list(candidate_options.keys()),
        key="recruiter_selected_candidate",
    )

    selected_candidate: CandidateResult = candidate_options[selected_label]

    with st.container():
        col_cand_info, col_cand_scores = st.columns([2, 1], gap="large")

        with col_cand_info:
            st.markdown(f"#### 👤 {selected_candidate.name}")
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                st.markdown(f"- ✉️ **Email:** `{selected_candidate.email or 'Not detected'}`")
                st.markdown(f"- 📞 **Phone:** `{selected_candidate.phone or 'Not detected'}`")
            with col_c2:
                st.markdown(f"- 🔗 **LinkedIn:** {selected_candidate.linkedin or 'Not detected'}")
                st.markdown(f"- 💻 **GitHub:** {selected_candidate.github or 'Not detected'}")

            exp_label = (
                f"{selected_candidate.years_of_experience:.1f} years"
                if selected_candidate.years_of_experience is not None
                else "Not explicitly stated"
            )
            st.markdown(f"- ⏳ **Experience Detected:** **{exp_label}**")

        with col_cand_scores:
            badge_color = "#10B981" if selected_candidate.composite_score >= pass_threshold else "#EF4444"
            st.markdown(
                f"""
                <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:12px; padding:1.2rem; text-align:center;">
                    <div style="font-size:2rem; font-weight:800; color:{badge_color};">{selected_candidate.composite_score}%</div>
                    <div style="font-size:0.85rem; color:#64748B; font-weight:600; text-transform:uppercase;">Composite Match Score</div>
                    <div style="margin-top:6px; font-size:0.8rem; color:#334155;">
                        Skill Match: <b>{selected_candidate.skill_match_percentage}%</b> | TF-IDF: <b>{selected_candidate.tfidf_similarity}%</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br/>", unsafe_allow_html=True)

        col_matched_box, col_missing_box = st.columns(2, gap="medium")
        with col_matched_box:
            st.markdown(f"##### ✅ Matched Required Skills ({len(selected_candidate.matched_skills)})")
            if selected_candidate.matched_skills:
                m_badges = " ".join([f'<span class="skill-badge skill-badge-green">✓ {s}</span>' for s in selected_candidate.matched_skills])
                st.markdown(m_badges, unsafe_allow_html=True)
            else:
                st.caption("No matching skills detected from job description.")

        with col_missing_box:
            st.markdown(f"##### ⚠️ Missing Required Skills ({len(selected_candidate.missing_skills)})")
            if selected_candidate.missing_skills:
                miss_badges = " ".join([f'<span class="skill-badge skill-badge-red">✗ {s}</span>' for s in selected_candidate.missing_skills])
                st.markdown(miss_badges, unsafe_allow_html=True)
            else:
                st.success("Candidate matches 100% of required skills!")

        # --- CHATGPT EVALUATION DOSSIER ---
        if selected_candidate.chatgpt_evaluation:
            cg = selected_candidate.chatgpt_evaluation
            st.markdown("<br/>", unsafe_allow_html=True)
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #1E1B4B 0%, #312E81 100%); border-radius: 12px; padding: 1.25rem 1.5rem; color: white; margin-bottom: 1.25rem;">
                    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
                        <div>
                            <span style="background:rgba(255,255,255,0.2); padding:3px 10px; border-radius:9999px; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.5px;">🤖 ChatGPT Recruiter Assessment</span>
                            <h4 style="margin:6px 0 2px 0; color:white;">{cg.verdict} &nbsp;•&nbsp; {cg.fit_level}</h4>
                        </div>
                        <div style="text-align:right;">
                            <span style="font-size:1.8rem; font-weight:800; color:#38BDF8;">{cg.chatgpt_score:.1f}%</span>
                            <div style="font-size:0.75rem; color:#CBD5E1;">AI Fit Score • {cg.source}</div>
                        </div>
                    </div>
                    <p style="margin: 10px 0 0 0; font-size:0.92rem; line-height:1.5; color:#E0E7FF;">
                        {cg.executive_summary}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            col_ai_strengths, col_ai_gaps = st.columns(2, gap="medium")
            with col_ai_strengths:
                st.markdown("##### 🌟 Candidate Strengths (ChatGPT)")
                for s in cg.key_strengths:
                    st.markdown(f"- ✅ {s}")

            with col_ai_gaps:
                st.markdown("##### ⚠️ Skill Gaps & Potential Risks (ChatGPT)")
                for g in cg.critical_gaps:
                    st.markdown(f"- ⚠️ {g}")

            st.markdown("<br/>", unsafe_allow_html=True)
            st.markdown("##### 📞 Tailored Technical Phone Screen Questions")
            st.caption("Custom technical screening questions formulated to test this candidate's specific qualification gaps:")
            for q_idx, q in enumerate(cg.interview_questions, 1):
                st.markdown(f"**Q{q_idx}:** *\"{q}\"*")

        if selected_candidate.key_bullet_points:
            st.markdown("<br/>", unsafe_allow_html=True)
            st.markdown("##### 📌 Extracted Key Achievements & Highlights")
            for b in selected_candidate.key_bullet_points:
                st.markdown(f"- {b}")

        col_full_text, col_prompt = st.columns(2, gap="medium")
        with col_full_text:
            with st.expander("📄 View Full Parsed Resume Text", expanded=False):
                st.text_area(
                    "Raw Text:",
                    value=selected_candidate.raw_text,
                    height=260,
                    disabled=True,
                    key=f"raw_text_{selected_candidate.filename}",
                )

        with col_prompt:
            with st.expander("📋 Copy Prompt for ChatGPT Web", expanded=False):
                cand_prompt = generate_chatgpt_prompt(selected_candidate.raw_text, jd_input)
                st.text_area(
                    "Paste into chatgpt.com:",
                    value=cand_prompt,
                    height=260,
                    disabled=False,
                    key=f"prompt_{selected_candidate.filename}",
                )
