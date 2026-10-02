import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from chatgpt_evaluator import (
    ChatGptEvaluationResult,
    are_skills_equivalent,
    classify_jd_requirements,
    emulate_chatgpt_evaluation,
    evaluate_resume_chatgpt,
    generate_chatgpt_prompt,
    match_skills_semantically,
)
from recruiter_mode import create_demo_resume_zip, process_resume_zip


SAMPLE_RESUME = """ALEX RIVERA
alex.rivera@email.com | +1 (555) 492-8172 | San Francisco, CA
linkedin.com/in/alexrivera-eng | github.com/alexrivera

PROFESSIONAL SUMMARY
Senior Full-Stack Engineer with 6+ years of experience building high-scale cloud platforms.
Specialized in Python, React, FastAPI, AWS microservices, and PostgreSQL.

WORK EXPERIENCE
Lead Software Engineer | Datacore Cloud (2021 - Present)
- Architected enterprise cloud services using Python, FastAPI, React, and PostgreSQL serving 15M+ requests daily.
- Deployed Docker containers to Kubernetes on AWS ECS, reducing deployment latency by 55%.
- Implemented Redis caching layers and GraphQL endpoints, improving API latency by 40%.

SKILLS
Languages: Python, TypeScript, JavaScript, SQL, Golang
Frameworks: React, Next.js, FastAPI, Node.js, GraphQL
Cloud & Tools: AWS, Docker, Kubernetes, Terraform, Redis, PostgreSQL, Git
"""

SAMPLE_JD = """Senior Full-Stack Engineer

Requirements:
- 4+ years of software development experience.
- Must have strong experience with Python, React, and AWS.
- Hands-on expertise with Docker, Kubernetes, and PostgreSQL.

Nice to Have & Bonus Qualifications:
- Familiarity with Golang and PyTorch is a plus.
- Experience with Kafka message queues.
"""


def test_are_skills_equivalent():
    # Canonical synonyms
    assert are_skills_equivalent("react", "react.js") is True
    assert are_skills_equivalent("React", "ReactJS") is True
    assert are_skills_equivalent("k8s", "kubernetes") is True
    assert are_skills_equivalent("golang", "go") is True
    assert are_skills_equivalent("postgres", "postgresql") is True
    assert are_skills_equivalent("ts", "typescript") is True
    assert are_skills_equivalent("docker", "containerization") is True

    # Non-equivalent skills
    assert are_skills_equivalent("python", "java") is False
    assert are_skills_equivalent("docker", "react") is False


def test_classify_jd_requirements():
    mandatory, preferred = classify_jd_requirements(SAMPLE_JD)

    # Core requirements should include Python, React, AWS, Docker, Kubernetes, PostgreSQL
    assert any(s in mandatory for s in ["Python", "React", "AWS", "Docker", "PostgreSQL"])

    # Preferred / bonus section should catch Golang, PyTorch, or Kafka
    assert any(s in preferred for s in ["Golang", "PyTorch", "Kafka"])


def test_match_skills_semantically():
    target_skills = ["React", "Kubernetes", "PostgreSQL", "Kafka"]
    candidate_skills = ["React.js", "K8s", "Postgres", "Redis"]

    matched, missing = match_skills_semantically(
        target_skills=target_skills,
        candidate_skills=candidate_skills,
        resume_text="Experienced with React.js, K8s, and Postgres database administration."
    )

    # Synonyms should match React, Kubernetes, and PostgreSQL
    assert "React" in matched
    assert "Kubernetes" in matched
    assert "PostgreSQL" in matched

    # Kafka was not present
    assert "Kafka" in missing


def test_emulate_chatgpt_evaluation():
    eval_res = emulate_chatgpt_evaluation(
        resume_text=SAMPLE_RESUME,
        jd_text=SAMPLE_JD,
        candidate_name="Alex Rivera",
    )

    assert isinstance(eval_res, ChatGptEvaluationResult)
    assert 60.0 <= eval_res.chatgpt_score <= 100.0
    assert eval_res.verdict in ["Strong Shortlist ✅", "Consider / Phone Screen ⚖️", "Pass / High Risk ❌"]
    assert eval_res.fit_level in ["Strong Fit", "Moderate Fit", "Weak Fit"]
    assert len(eval_res.executive_summary) > 20

    # Strengths and gaps
    assert len(eval_res.key_strengths) >= 1
    assert len(eval_res.critical_gaps) >= 1

    # Interview questions
    assert len(eval_res.interview_questions) >= 2
    for q in eval_res.interview_questions:
        assert len(q) > 10

    # Bullet rewrites (X-Y-Z formula)
    assert len(eval_res.bullet_rewrites) >= 1
    assert "original" in eval_res.bullet_rewrites[0]
    assert "rewrite" in eval_res.bullet_rewrites[0]


def test_generate_chatgpt_prompt():
    prompt = generate_chatgpt_prompt(SAMPLE_RESUME, SAMPLE_JD)
    assert "TARGET JOB DESCRIPTION:" in prompt
    assert "CANDIDATE RESUME:" in prompt
    assert "Google X-Y-Z" in prompt
    assert "Senior Full-Stack Engineer" in prompt
    assert "Alex Rivera" in prompt or "ALEX RIVERA" in prompt


def test_evaluate_resume_chatgpt_fallback():
    # Providing no key or invalid key should seamlessly fall back to the emulator without crashing
    res = evaluate_resume_chatgpt(
        resume_text=SAMPLE_RESUME,
        jd_text=SAMPLE_JD,
        candidate_name="Alex Rivera",
        api_key=None,
    )

    assert res is not None
    assert res.chatgpt_score > 50.0
    assert res.source == "Built-in ChatGPT Semantic Emulator"


def test_process_resume_zip_populates_chatgpt_evaluation():
    zip_bytes = create_demo_resume_zip()
    results = process_resume_zip(
        zip_bytes=zip_bytes,
        jd_text=SAMPLE_JD,
        threshold=60.0,
    )

    for c in results.candidates:
        if c.status == "Success":
            assert c.chatgpt_evaluation is not None
            assert isinstance(c.chatgpt_evaluation, ChatGptEvaluationResult)
            assert len(c.chatgpt_evaluation.interview_questions) > 0
            assert len(c.chatgpt_evaluation.executive_summary) > 0
