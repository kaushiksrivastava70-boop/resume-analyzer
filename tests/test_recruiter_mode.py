import io
import sys
import zipfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pytest
from recruiter_mode import (
    BatchAnalysisResult,
    CandidateResult,
    SimpleTfidfCosineMatcher,
    candidates_to_dataframe,
    clean_filename_to_name,
    compute_tfidf_similarities,
    create_demo_resume_zip,
    export_to_csv,
    export_to_excel,
    extract_bullet_points,
    extract_candidate_name,
    extract_years_of_experience,
    process_resume_zip,
)


SAMPLE_RESUME_TEXT = """JOHNATHAN DOE
john.doe@techpro.com | +1 (555) 987-6543 | Chicago, IL
linkedin.com/in/johndoe-pro | github.com/johndoe-code

PROFESSIONAL SUMMARY
Senior Software Architect with 8+ years of experience engineering cloud-native microservices.
Led engineering teams to modernize legacy monoliths into distributed Kubernetes applications.

WORK EXPERIENCE
Lead Engineer | Nexus Cloud Inc (2020 - Present)
- Architected high-throughput microservices using Python, FastAPI, and PostgreSQL, reducing latency by 45%.
- Led migration of 50+ services to Docker and AWS ECS with Terraform.
- Implemented Redis distributed caching, serving 20M+ daily requests with 99.99% uptime.

Senior Developer | CodeWorks (2016 - 2020)
- Developed REST APIs in Python and Django with MySQL.

TECHNICAL SKILLS
Languages: Python, Go, TypeScript, SQL
Cloud: AWS, Docker, Kubernetes, Terraform
Databases: PostgreSQL, Redis, MySQL
Frameworks: FastAPI, Django, React
"""

SAMPLE_JD_TEXT = """Senior Cloud & Python Engineer
Requirements:
- 5+ years experience building cloud applications.
- Strong knowledge of Python, FastAPI, Docker, and AWS.
- Hands-on expertise with PostgreSQL, Redis, and Kubernetes.
- Experience with React is a plus.
"""


def test_clean_filename_to_name():
    assert clean_filename_to_name("john_doe_resume_2024.pdf") == "John Doe"
    assert clean_filename_to_name("Jane-Smith-CV-Updated.docx") == "Jane Smith"
    assert clean_filename_to_name("Alex_Rivera_Senior_FullStack.txt") == "Alex Rivera Senior"
    assert clean_filename_to_name("resume.pdf") == "Candidate"


def test_extract_candidate_name():
    name = extract_candidate_name(SAMPLE_RESUME_TEXT, "john_doe_resume.pdf")
    assert name == "Johnathan Doe"

    # Fallback when no valid name line present
    fallback_name = extract_candidate_name("random text without a proper name", "sarah_connor_resume.pdf")
    assert "Sarah Connor" in fallback_name


def test_extract_years_of_experience():
    exp1 = extract_years_of_experience(SAMPLE_RESUME_TEXT)
    assert exp1 == 8.0

    exp2 = extract_years_of_experience("Over 3.5 years of hands-on experience in machine learning.")
    assert exp2 == 3.5

    exp3 = extract_years_of_experience("Software Developer with five years of experience.")
    assert exp3 == 5.0

    # Year range detection fallback
    exp4 = extract_years_of_experience("Software Engineer at Google (2018 - 2024)")
    assert exp4 == 6.0


def test_extract_bullet_points():
    bullets = extract_bullet_points(SAMPLE_RESUME_TEXT, max_bullets=3)
    assert len(bullets) > 0
    assert any("microservices" in b.lower() or "latency" in b.lower() for b in bullets)


def test_tfidf_cosine_matcher():
    jd = "Python FastAPI AWS Docker PostgreSQL"
    resume1 = "Expert in Python, FastAPI, AWS, Docker, and PostgreSQL with high scalability."
    resume2 = "Chef specializing in Italian cuisine, pasta, pizza, and pastry baking."

    scores = compute_tfidf_similarities(jd, [resume1, resume2])
    assert len(scores) == 2
    assert scores[0] > scores[1]
    assert scores[0] > 40.0
    assert scores[1] < 10.0


def test_create_demo_resume_zip():
    zip_bytes = create_demo_resume_zip()
    assert len(zip_bytes) > 0

    # Verify it is a valid zip containing 5 resumes
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        namelist = zf.namelist()
        assert len(namelist) == 5
        assert "Alex_Rivera_Senior_FullStack.txt" in namelist


def test_process_resume_zip():
    zip_bytes = create_demo_resume_zip()
    result = process_resume_zip(
        zip_bytes=zip_bytes,
        jd_text=SAMPLE_JD_TEXT,
        threshold=60.0,
    )

    assert isinstance(result, BatchAnalysisResult)
    assert result.total_scanned == 5
    assert len(result.candidates) == 5

    # Verify candidates are sorted by composite score descending
    for i in range(len(result.candidates) - 1):
        assert result.candidates[i].composite_score >= result.candidates[i + 1].composite_score

    # Alex Rivera should rank near the top for FullStack/Python/AWS/Docker/Postgres/Redis
    top_candidate = result.candidates[0]
    assert "Alex" in top_candidate.name or "Priya" in top_candidate.name
    assert top_candidate.composite_score > 60.0
    assert top_candidate.email is not None

    # Check analytics
    assert result.shortlisted_count >= 1
    assert result.average_score > 0
    assert len(result.top_missing_skills) > 0


def test_process_resume_zip_corrupt_file_handling():
    # Create zip with one valid resume and one corrupted file
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("valid_candidate.txt", SAMPLE_RESUME_TEXT)
        zf.writestr("corrupted_candidate.pdf", b"NOT A VALID PDF HEADER CORRUPT BYTES")

    result = process_resume_zip(
        zip_bytes=buf.getvalue(),
        jd_text=SAMPLE_JD_TEXT,
        threshold=60.0,
    )

    # Should process valid candidate and gracefully flag the corrupt file
    assert result.total_scanned == 2
    valid_res = [c for c in result.candidates if c.status == "Success"]
    err_res = [c for c in result.candidates if c.status == "Error"]

    assert len(valid_res) == 1
    assert len(err_res) == 1
    assert "corrupted_candidate.pdf" in err_res[0].filename
    assert len(result.errors) == 1


def test_candidates_to_dataframe_and_exports():
    zip_bytes = create_demo_resume_zip()
    result = process_resume_zip(
        zip_bytes=zip_bytes,
        jd_text=SAMPLE_JD_TEXT,
        threshold=60.0,
    )

    df = candidates_to_dataframe(result.candidates, threshold=60.0)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5
    assert "Rank" in df.columns
    assert "Status" in df.columns
    assert "Candidate Name" in df.columns
    assert "Match Score (%)" in df.columns

    # Test CSV export
    csv_bytes = export_to_csv(df)
    assert len(csv_bytes) > 0
    assert b"Candidate Name" in csv_bytes

    # Test Excel export
    excel_bytes = export_to_excel(df)
    assert len(excel_bytes) > 0
    # Read back with pandas/openpyxl
    df_read = pd.read_excel(io.BytesIO(excel_bytes), sheet_name="Candidate Rankings")
    assert len(df_read) == 5
    assert "Candidate Name" in df_read.columns


def test_process_resume_zip_empty_and_invalid():
    # Empty bytes
    with pytest.raises(ValueError, match="empty"):
        process_resume_zip(b"", SAMPLE_JD_TEXT)

    # Bad zip file
    with pytest.raises(ValueError, match="not a valid ZIP archive"):
        process_resume_zip(b"invalid zip content", SAMPLE_JD_TEXT)

    # Zip with no supported files
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("image.png", b"fake png data")
        zf.writestr("data.csv", b"a,b,c\n1,2,3")
    with pytest.raises(ValueError, match="No supported resume files"):
        process_resume_zip(buf.getvalue(), SAMPLE_JD_TEXT)
