import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from analyzer import (
    ContactInfo,
    ResumeExtractionError,
    analyze_resume,
    calculate_readability,
    extract_action_verbs,
    extract_contact_info,
    extract_quantified_metrics,
    extract_sections,
    extract_skills,
    generate_pdf_report,
    generate_text_report,
    match_job_description,
    score_resume,
)


SAMPLE_TEXT = """
ALEX MORGAN
alex.morgan@email.com | +1 (555) 382-9104 | San Francisco, CA
linkedin.com/in/alexmorgan-dev | github.com/alexmorgan-tech

PROFESSIONAL SUMMARY
Experienced Senior Full-Stack Engineer with 5+ years building distributed cloud systems.
Reduced system latency by 42% and achieved 99.99% uptime across 10M+ users.

WORK EXPERIENCE
Senior Software Engineer | CloudScale Systems
- Architected microservices with Python, FastAPI, and PostgreSQL, increasing throughput by 65%.
- Deployed Docker containers to AWS ECS, saving $45,000/yr in cloud spend.
- Optimized query execution with Redis caching, dropping response time to 18ms.

TECHNICAL SKILLS
Languages: Python, TypeScript, JavaScript, SQL
Frameworks: React, Next.js, FastAPI, Node.js
Cloud & Tools: AWS, Docker, Kubernetes, Git, PostgreSQL, Redis

EDUCATION
Bachelor of Science in Computer Science
UC Berkeley | 2015 - 2019
"""


def test_extract_contact_info():
    info = extract_contact_info(SAMPLE_TEXT)
    assert info.email == "alex.morgan@email.com"
    assert "555" in info.phone
    assert "linkedin.com/in/alexmorgan-dev" in info.linkedin
    assert "github.com/alexmorgan-tech" in info.github
    assert info.completeness_score() == 15.0


def test_extract_sections():
    sections = extract_sections(SAMPLE_TEXT)
    assert sections["Work Experience"] is True
    assert sections["Skills"] is True
    assert sections["Education"] is True
    assert sections["Summary / Objective"] is True


def test_extract_skills():
    by_cat, all_skills = extract_skills(SAMPLE_TEXT)
    assert "Python" in all_skills
    assert "FastAPI" in all_skills
    assert "AWS" in all_skills
    assert "Docker" in all_skills
    assert "PostgreSQL" in all_skills
    assert "React" in all_skills
    assert "TypeScript" in all_skills

    # Category checks
    assert "Python" in by_cat["Programming Languages"]
    assert "AWS" in by_cat["Cloud & DevOps"]
    assert "PostgreSQL" in by_cat["Databases & Tools"]


def test_extract_action_verbs():
    verbs = extract_action_verbs(SAMPLE_TEXT)
    assert "architected" in verbs or "built" in verbs or "optimized" in verbs or "achieved" in verbs


def test_extract_quantified_metrics():
    metrics = extract_quantified_metrics(SAMPLE_TEXT)
    assert any("42%" in m for m in metrics)
    assert any("99.99%" in m for m in metrics)
    assert any("65%" in m for m in metrics)
    assert any("45,000" in m or "$45" in m for m in metrics)


def test_readability_and_scoring():
    words, sentences, fre = calculate_readability(SAMPLE_TEXT)
    assert words > 80
    assert sentences > 5
    assert 0.0 <= fre <= 100.0

    analysis = analyze_resume(SAMPLE_TEXT)
    assert 50.0 <= analysis.total_score <= 100.0
    assert len(analysis.suggestions) > 0


def test_job_description_matching():
    jd = """
    We are looking for a Senior Python Developer with deep experience in AWS,
    Docker, Kubernetes, and Golang. Knowledge of GraphQL and PyTorch is a plus.
    """
    res = match_job_description(SAMPLE_TEXT, jd)
    assert res.jd_skills_count > 0
    assert "Python" in res.matched_skills
    assert "AWS" in res.matched_skills
    assert "Docker" in res.matched_skills
    assert "Kubernetes" in res.matched_skills
    assert "Golang" in res.missing_skills or "PyTorch" in res.missing_skills
    assert 0.0 < res.match_percentage <= 100.0


def test_empty_input_handling():
    with pytest.raises(ResumeExtractionError):
        analyze_resume("")

    with pytest.raises(ResumeExtractionError):
        analyze_resume("    \n\t   ")


def test_report_generation():
    analysis = analyze_resume(SAMPLE_TEXT)
    txt_report = generate_text_report(analysis)
    assert "RESUME ANALYZER AUDIT REPORT" in txt_report
    assert "alex.morgan@email.com" in txt_report

    pdf_bytes = generate_pdf_report(analysis)
    assert len(pdf_bytes) > 500
    assert pdf_bytes.startswith(b"%PDF")


def test_pdf_and_docx_extraction():
    import io
    from reportlab.pdfgen import canvas
    from docx import Document
    from analyzer import extract_text_from_pdf, extract_text_from_docx, extract_text

    # Test PDF extraction
    pdf_buffer = io.BytesIO()
    c = canvas.Canvas(pdf_buffer)
    c.drawString(100, 750, "Alex Morgan Software Engineer alex.morgan@email.com with experience in Python and AWS")
    c.save()
    pdf_bytes = pdf_buffer.getvalue()

    pdf_extracted = extract_text_from_pdf(pdf_bytes)
    assert "Alex Morgan" in pdf_extracted
    assert "Python" in pdf_extracted

    # Test DOCX extraction
    docx_buffer = io.BytesIO()
    doc = Document()
    doc.add_paragraph("Alex Morgan Software Engineer alex.morgan@email.com with experience in Python and AWS")
    doc.save(docx_buffer)
    docx_bytes = docx_buffer.getvalue()

    docx_extracted = extract_text_from_docx(docx_bytes)
    assert "Alex Morgan" in docx_extracted
    assert "Python" in docx_extracted

    # Test generic extract_text dispatcher
    assert "Python" in extract_text(pdf_bytes, "test.pdf")
    assert "Python" in extract_text(docx_bytes, "test.docx")
    assert "Python" in extract_text(b"Python developer resume text", "test.txt")

