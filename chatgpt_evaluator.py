"""ChatGPT-Grade Semantic Resume Evaluator & Comparison Engine.

Provides deep semantic analysis mimicking ChatGPT / OpenAI models:
- Concept and synonym mapping (e.g. k8s <-> kubernetes, postgres <-> sql/relational)
- Classification of Mandatory vs. Preferred / Bonus requirements in Job Descriptions
- Experience & Seniority alignment scoring
- Live OpenAI API connectivity (gpt-4o-mini / gpt-4o) via lightweight REST calls
- Built-in deterministic ChatGPT Semantic Emulator (100% offline, zero-cost fallback)
- Tailored technical phone screen questions targeting candidate skill gaps
- Google X-Y-Z achievement bullet point rewrites
- 1-click ChatGPT Web prompt generation
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import requests

from analyzer import extract_skills


@dataclass
class ChatGptEvaluationResult:
    """Stores structured qualitative and quantitative evaluation aligned with ChatGPT."""
    chatgpt_score: float
    verdict: str  # "Strong Shortlist ✅", "Consider / Phone Screen ⚖️", "Pass / High Risk ❌"
    fit_level: str  # "Strong Fit", "Moderate Fit", "Weak Fit"
    executive_summary: str
    key_strengths: List[str]
    critical_gaps: List[str]
    interview_questions: List[str]
    bullet_rewrites: List[Dict[str, str]]  # [{"original": ..., "rewrite": ...}]
    mandatory_matched: List[str]
    mandatory_missing: List[str]
    preferred_matched: List[str]
    preferred_missing: List[str]
    source: str = "Built-in ChatGPT Semantic Emulator"
    raw_response: Optional[str] = None


# =========================================================================
# 1. SEMANTIC CONCEPT & SYNONYM TAXONOMY
# =========================================================================

# Maps variants, abbreviations, and related technologies to canonical concepts
SYNONYM_GROUPS: List[Set[str]] = [
    {"react", "react.js", "reactjs", "react-native"},
    {"vue", "vue.js", "vuejs"},
    {"angular", "angularjs", "angular 2+"},
    {"node", "node.js", "nodejs", "express", "express.js"},
    {"python", "python3", "py"},
    {"fastapi", "flask", "django"},
    {"typescript", "ts"},
    {"javascript", "js", "ecmascript", "es6"},
    {"golang", "go"},
    {"c#", "csharp", ".net", "dotnet", "asp.net"},
    {"c++", "cpp"},
    {"postgres", "postgresql", "psql"},
    {"mongo", "mongodb"},
    {"k8s", "kubernetes"},
    {"docker", "containerization", "containers"},
    {"aws", "amazon web services", "amazon ec2", "aws ecs", "aws s3", "aws lambda"},
    {"gcp", "google cloud platform", "google cloud"},
    {"azure", "microsoft azure"},
    {"ci/cd", "cicd", "continuous integration", "continuous deployment", "github actions", "gitlab ci", "jenkins"},
    {"terraform", "infrastructure as code", "iac"},
    {"kafka", "apache kafka", "message broker", "rabbitmq"},
    {"redis", "in-memory cache", "memcached"},
    {"graphql", "rest", "restful", "rest api", "grpc"},
    {"sql", "relational database", "rdbms", "mysql", "postgresql", "oracle", "sqlite"},
    {"nosql", "dynamodb", "mongodb", "cassandra", "couchdb"},
    {"pytorch", "tensorflow", "keras", "deep learning", "machine learning", "ml"},
    {"microservices", "distributed systems", "service-oriented architecture", "soa"},
    {"agile", "scrum", "kanban", "sprints"},
]

# Quick lookup dict from token to set of group indices
TOKEN_TO_GROUPS: Dict[str, Set[int]] = {}
for group_idx, grp in enumerate(SYNONYM_GROUPS):
    for token in grp:
        t_low = token.lower()
        if t_low not in TOKEN_TO_GROUPS:
            TOKEN_TO_GROUPS[t_low] = set()
        TOKEN_TO_GROUPS[t_low].add(group_idx)


def are_skills_equivalent(skill_a: str, skill_b: str) -> bool:
    """Determine if two skills are semantically equivalent or closely aligned."""
    a_low = skill_a.strip().lower()
    b_low = skill_b.strip().lower()

    if a_low == b_low:
        return True

    groups_a = TOKEN_TO_GROUPS.get(a_low, set())
    groups_b = TOKEN_TO_GROUPS.get(b_low, set())

    if groups_a and groups_b and (groups_a & groups_b):
        return True

    # Fallback substring checks for compound terms (e.g. "React" in "React.js Developer")
    if len(a_low) > 3 and a_low in b_low:
        return True
    if len(b_low) > 3 and b_low in a_low:
        return True

    return False


# =========================================================================
# 2. JOB DESCRIPTION REQUIREMENT CLASSIFICATION
# =========================================================================

def classify_jd_requirements(jd_text: str) -> Tuple[List[str], List[str]]:
    """Classify JD skills into Mandatory (Core) vs. Preferred (Bonus/Plus).

    Mimics ChatGPT's understanding that missing a 'must-have' skill is far more
    consequential than missing a 'nice-to-have' skill.
    """
    _, all_jd_skills = extract_skills(jd_text)
    if not all_jd_skills:
        return [], []

    # Split JD text into sentences or lines
    lines = [l.strip() for l in re.split(r"[\n\r]+", jd_text) if l.strip()]

    mandatory_skills: Set[str] = set()
    preferred_skills: Set[str] = set()

    # Preferred / bonus indicators
    preferred_markers = [
        "plus", "bonus", "nice to have", "preferred", "advantageous", "desired",
        "optional", "good to have", "familiarity with", "ideally", "helpful"
    ]

    # Mandatory / required indicators
    mandatory_markers = [
        "require", "must have", "minimum", "essential", "qualifications", "proficien",
        "hands-on experience with", "expected", "core", "proven track record in"
    ]

    current_section = "mandatory"  # Default assumption for job postings

    for line in lines:
        line_lower = line.lower()

        # Check for section headers
        if any(pm in line_lower for pm in ["preferred qualifications", "nice to have", "bonus points", "bonus qualifications"]):
            current_section = "preferred"
            continue
        elif any(mm in line_lower for mm in ["required qualifications", "minimum qualifications", "requirements", "what you need"]):
            current_section = "mandatory"
            continue

        # Check line-level markers
        is_preferred_line = any(pm in line_lower for pm in preferred_markers)
        is_mandatory_line = any(mm in line_lower for mm in mandatory_markers)

        for skill in all_jd_skills:
            # Check if skill is in this line
            if re.search(rf"\b{re.escape(skill)}\b", line, re.IGNORECASE):
                if is_preferred_line or current_section == "preferred":
                    preferred_skills.add(skill)
                else:
                    mandatory_skills.add(skill)

    # Any skill not explicitly categorized as preferred defaults to mandatory
    unassigned = set(all_jd_skills) - mandatory_skills - preferred_skills
    mandatory_skills.update(unassigned)

    # If everything got assigned to preferred (rare edge case), assign back to mandatory
    if not mandatory_skills and preferred_skills:
        mandatory_skills = preferred_skills
        preferred_skills = set()

    return sorted(list(mandatory_skills)), sorted(list(preferred_skills))


def match_skills_semantically(
    target_skills: List[str], candidate_skills: List[str], resume_text: str
) -> Tuple[List[str], List[str]]:
    """Match skills accounting for synonyms, concept groups, and raw text context."""
    matched: List[str] = []
    missing: List[str] = []

    candidate_skills_lower = [s.lower() for s in candidate_skills]
    resume_lower = resume_text.lower()

    for target in target_skills:
        found = False

        # Direct match in extracted skills
        for cs in candidate_skills:
            if are_skills_equivalent(target, cs):
                matched.append(target)
                found = True
                break

        # Check raw resume text with regex / boundary
        if not found:
            t_low = target.lower()
            if are_skills_equivalent(target, t_low) and re.search(rf"\b{re.escape(t_low)}\b", resume_lower):
                matched.append(target)
                found = True

        # Check synonym group members in raw text
        if not found and target.lower() in TOKEN_TO_GROUPS:
            for group_idx in TOKEN_TO_GROUPS[target.lower()]:
                for member in SYNONYM_GROUPS[group_idx]:
                    if re.search(rf"\b{re.escape(member)}\b", resume_lower):
                        matched.append(target)
                        found = True
                        break
                if found:
                    break

        if not found:
            missing.append(target)

    return sorted(list(set(matched))), sorted(list(set(missing)))


# =========================================================================
# 3. CHATGPT SEMANTIC EMULATOR (100% OFFLINE / ZERO-COST FALLBACK)
# =========================================================================

def emulate_chatgpt_evaluation(
    resume_text: str,
    jd_text: str,
    candidate_name: str = "Candidate",
) -> ChatGptEvaluationResult:
    """Deterministic, high-fidelity semantic evaluation mimicking ChatGPT's reasoning."""
    if not resume_text.strip():
        return ChatGptEvaluationResult(
            chatgpt_score=0.0,
            verdict="Pass / High Risk ❌",
            fit_level="Weak Fit",
            executive_summary="Empty resume text provided.",
            key_strengths=[],
            critical_gaps=["No readable content found in resume."],
            interview_questions=[],
            bullet_rewrites=[],
            mandatory_matched=[],
            mandatory_missing=[],
            preferred_matched=[],
            preferred_missing=[],
        )

    # 1. Classify JD requirements
    mandatory_reqs, preferred_reqs = classify_jd_requirements(jd_text)
    _, candidate_skills = extract_skills(resume_text)

    # 2. Semantic matching with synonym support
    mand_matched, mand_missing = match_skills_semantically(mandatory_reqs, candidate_skills, resume_text)
    pref_matched, pref_missing = match_skills_semantically(preferred_reqs, candidate_skills, resume_text)

    # 3. Calculate weighted skill score: Mandatory = 75%, Preferred = 25%
    mand_rate = (len(mand_matched) / max(1, len(mandatory_reqs))) if mandatory_reqs else 1.0
    pref_rate = (len(pref_matched) / max(1, len(preferred_reqs))) if preferred_reqs else 1.0

    if mandatory_reqs and preferred_reqs:
        weighted_skill_score = (mand_rate * 0.75) + (pref_rate * 0.25)
    elif mandatory_reqs:
        weighted_skill_score = mand_rate
    else:
        weighted_skill_score = 0.70

    # 4. Experience & Seniority alignment
    from recruiter_mode import extract_years_of_experience
    cand_exp = extract_years_of_experience(resume_text) or 2.0
    jd_exp_match = re.search(r"\b(\d+)\+?\s*(?:years?|yrs?)\b", jd_text, re.IGNORECASE)
    req_exp = float(jd_exp_match.group(1)) if jd_exp_match else 3.0

    exp_ratio = min(1.2, cand_exp / max(1.0, req_exp))
    exp_factor = min(1.0, exp_ratio)

    # 5. Impact & Metric density
    has_metrics = bool(re.search(r"\b(\d+%|\$\d+|[kKmMbB]\+?\s*(?:users|requests))\b", resume_text))
    metric_bonus = 0.05 if has_metrics else 0.0

    # 6. Overall ChatGPT-grade Score (0-100)
    raw_score = (weighted_skill_score * 0.75 + exp_factor * 0.20 + metric_bonus) * 100.0
    final_score = round(max(5.0, min(98.0, raw_score)), 1)

    # 7. Formulate Verdict & Fit Level
    if final_score >= 75.0 and len(mand_missing) <= 1:
        verdict = "Strong Shortlist ✅"
        fit_level = "Strong Fit"
    elif final_score >= 55.0:
        verdict = "Consider / Phone Screen ⚖️"
        fit_level = "Moderate Fit"
    else:
        verdict = "Pass / High Risk ❌"
        fit_level = "Weak Fit"

    # 8. Generate Contextual Executive Summary
    summary_parts = []
    if fit_level == "Strong Fit":
        summary_parts.append(
            f"{candidate_name} represents a high-conviction match for the role, demonstrating proven alignment with "
            f"core stack requirements ({', '.join(mand_matched[:4])})."
        )
    elif fit_level == "Moderate Fit":
        summary_parts.append(
            f"{candidate_name} shows solid transferable capabilities, matching several key technologies ({', '.join(mand_matched[:3]) if mand_matched else 'general software'}), "
            f"but has notable gaps in target competencies that warrant technical screening."
        )
    else:
        summary_parts.append(
            f"{candidate_name}'s background diverges substantially from the required stack and seniority expectations for this role."
        )

    if cand_exp:
        summary_parts.append(f"Detected roughly {cand_exp:.1f} years of relevant experience against a {req_exp:.0f}+ year baseline.")

    executive_summary = " ".join(summary_parts)

    # 9. Key Strengths (3-4 points)
    key_strengths = []
    if mand_matched:
        key_strengths.append(f"Strong overlap in core mandatory requirements: {', '.join(mand_matched[:5])}.")
    if pref_matched:
        key_strengths.append(f"Bonus qualifications satisfied: {', '.join(pref_matched[:3])}.")
    if cand_exp >= req_exp:
        key_strengths.append(f"Meets or exceeds the required seniority threshold ({cand_exp:.1f} yrs detected).")
    if has_metrics:
        key_strengths.append("Demonstrates quantified business impact (percentages, revenue, or scale metrics present).")
    if not key_strengths:
        key_strengths.append("Foundational technical literacy in modern software development.")

    # 10. Critical Gaps & Risks
    critical_gaps = []
    if mand_missing:
        critical_gaps.append(f"Missing core mandatory qualifications: {', '.join(mand_missing[:4])}.")
    if pref_missing:
        critical_gaps.append(f"Lacks preferred/bonus domain tools: {', '.join(pref_missing[:3])}.")
    if cand_exp < req_exp:
        critical_gaps.append(f"Seniority gap: estimated {cand_exp:.1f} years vs. target {req_exp:.0f}+ years requested.")
    if not has_metrics:
        critical_gaps.append("Resume contains few or no quantified outcome metrics (mostly duty descriptions).")
    if not critical_gaps:
        critical_gaps.append("No critical technical blockers detected relative to the job posting.")

    # 11. Role-Specific Technical Interview Questions
    interview_questions = []
    # Q1: Target the top missing core skill
    if mand_missing:
        skill = mand_missing[0]
        interview_questions.append(
            f"Given our role requires hands-on experience with {skill}, can you walk us through any production exposure or architectural understanding you have in this area?"
        )
    else:
        interview_questions.append(
            f"Can you walk us through the most technically challenging system you designed using {mand_matched[0] if mand_matched else 'your primary stack'} and how you optimized its throughput?"
        )

    # Q2: Target scale / architecture / second gap
    if len(mand_missing) > 1:
        skill2 = mand_missing[1]
        interview_questions.append(
            f"How would you approach integrating {skill2} into our current tech stack, and what design trade-offs would you evaluate?"
        )
    elif pref_missing:
        interview_questions.append(
            f"We use {pref_missing[0]} for our operations. What related paradigms have you worked with, and how quickly could you ramp up?"
        )
    else:
        interview_questions.append(
            "How do you ensure zero-downtime deployments and handle automated failover in your microservices?"
        )

    # Q3: Production incident or leadership question
    interview_questions.append(
        "Describe a major production outage or performance degradation you diagnosed. What debugging strategy did you use, and how did you resolve it?"
    )

    # 12. Bullet Point Rewrites (Google X-Y-Z formula)
    bullet_rewrites = []
    from recruiter_mode import extract_bullet_points
    bullets = extract_bullet_points(resume_text, max_bullets=2)

    for b in bullets:
        # Transform bullet into high-impact X-Y-Z version
        clean_b = re.sub(r"^[A-Z][a-z]+ed\s+", "", b).strip()
        rewrite = f"Spearheaded {clean_b[:80].rstrip('.')}, improving system throughput by 35% and reducing incident latency across core microservices."
        bullet_rewrites.append({
            "original": b,
            "rewrite": rewrite,
        })

    if not bullet_rewrites:
        bullet_rewrites.append({
            "original": "Responsible for developing backend web services and collaborating with the team.",
            "rewrite": "Architected and deployed 4+ REST microservices using Python and PostgreSQL, reducing endpoint response times by 42% and supporting 1M+ active users.",
        })

    return ChatGptEvaluationResult(
        chatgpt_score=final_score,
        verdict=verdict,
        fit_level=fit_level,
        executive_summary=executive_summary,
        key_strengths=key_strengths,
        critical_gaps=critical_gaps,
        interview_questions=interview_questions,
        bullet_rewrites=bullet_rewrites,
        mandatory_matched=mand_matched,
        mandatory_missing=mand_missing,
        preferred_matched=pref_matched,
        preferred_missing=pref_missing,
        source="Built-in ChatGPT Semantic Emulator",
    )


# =========================================================================
# 4. LIVE OPENAI API CALLER
# =========================================================================

def evaluate_with_openai_api(
    resume_text: str,
    jd_text: str,
    api_key: str,
    model: str = "gpt-4o-mini",
    candidate_name: str = "Candidate",
) -> Optional[ChatGptEvaluationResult]:
    """Call OpenAI's Chat Completions REST API for live GPT evaluation."""
    if not api_key or not api_key.strip():
        return None

    system_prompt = (
        "You are an elite Silicon Valley Technical Recruiter & Hiring Bar Raiser. "
        "Evaluate the candidate resume against the target Job Description with ruthless objectivity. "
        "Return ONLY a valid JSON object matching this exact schema:\n"
        "{\n"
        '  "chatgpt_score": <number between 0 and 100>,\n'
        '  "verdict": "<Strong Shortlist ✅ | Consider / Phone Screen ⚖️ | Pass / High Risk ❌>",\n'
        '  "fit_level": "<Strong Fit | Moderate Fit | Weak Fit>",\n'
        '  "executive_summary": "<2-3 sentence executive assessment>",\n'
        '  "key_strengths": ["<strength 1>", "<strength 2>", "<strength 3>"],\n'
        '  "critical_gaps": ["<gap 1>", "<gap 2>", "<gap 3>"],\n'
        '  "interview_questions": ["<question 1>", "<question 2>", "<question 3>"],\n'
        '  "bullet_rewrites": [\n'
        '    {"original": "<weak bullet from resume>", "rewrite": "<Google X-Y-Z rewritten bullet>"}\n'
        "  ]\n"
        "}"
    )

    user_prompt = (
        f"CANDIDATE NAME: {candidate_name}\n\n"
        f"JOB DESCRIPTION:\n{jd_text[:3000]}\n\n"
        f"RESUME TEXT:\n{resume_text[:4000]}"
    )

    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30,
        )
        if response.status_code == 200:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            parsed = json.loads(content)

            # Extract mandatory/preferred for completeness
            mand_reqs, pref_reqs = classify_jd_requirements(jd_text)
            _, c_skills = extract_skills(resume_text)
            m_matched, m_missing = match_skills_semantically(mand_reqs, c_skills, resume_text)
            p_matched, p_missing = match_skills_semantically(pref_reqs, c_skills, resume_text)

            return ChatGptEvaluationResult(
                chatgpt_score=float(parsed.get("chatgpt_score", 70.0)),
                verdict=parsed.get("verdict", "Consider / Phone Screen ⚖️"),
                fit_level=parsed.get("fit_level", "Moderate Fit"),
                executive_summary=parsed.get("executive_summary", "Evaluation complete."),
                key_strengths=parsed.get("key_strengths", []),
                critical_gaps=parsed.get("critical_gaps", []),
                interview_questions=parsed.get("interview_questions", []),
                bullet_rewrites=parsed.get("bullet_rewrites", []),
                mandatory_matched=m_matched,
                mandatory_missing=m_missing,
                preferred_matched=p_matched,
                preferred_missing=p_missing,
                source=f"Live OpenAI API ({model})",
                raw_response=content,
            )
        else:
            return None
    except Exception:
        return None


def evaluate_resume_chatgpt(
    resume_text: str,
    jd_text: str,
    candidate_name: str = "Candidate",
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
) -> ChatGptEvaluationResult:
    """Orchestrate evaluation: uses Live OpenAI API if key available, else falls back to emulator."""
    # Check provided key or environment variable
    effective_key = api_key or os.environ.get("OPENAI_API_KEY")

    if effective_key:
        api_result = evaluate_with_openai_api(
            resume_text=resume_text,
            jd_text=jd_text,
            api_key=effective_key,
            model=model,
            candidate_name=candidate_name,
        )
        if api_result:
            return api_result

    # Fallback to high-fidelity semantic emulator
    return emulate_chatgpt_evaluation(
        resume_text=resume_text,
        jd_text=jd_text,
        candidate_name=candidate_name,
    )


# =========================================================================
# 5. 1-CLICK CHATGPT WEB PROMPT GENERATOR
# =========================================================================

def generate_chatgpt_prompt(resume_text: str, jd_text: str) -> str:
    """Generate a production-grade recruiter prompt ready to paste into ChatGPT web."""
    return f"""Act as a Principal Technical Recruiter and Engineering Bar Raiser at a Tier-1 tech company.

Perform a thorough, objective evaluation of the following Candidate Resume against the target Job Description.

====================
TARGET JOB DESCRIPTION:
====================
{jd_text.strip()}

====================
CANDIDATE RESUME:
====================
{resume_text.strip()}

====================
YOUR TASK:
====================
Provide a structured assessment covering:
1. Overall Fit Score (0 to 100%) and Verdict (Strong Shortlist | Consider / Screen | Pass)
2. Executive Summary (2-3 sentences evaluating stack match and seniority)
3. Top 3 Strengths for this specific role
4. Critical Skill Gaps & Potential Red Flags
5. 3 Custom Technical Phone Screen Questions to test candidate gaps
6. Google X-Y-Z Rewrite: Take 2 weak bullets from the resume and rewrite them into high-impact metrics (Accomplished [X] as measured by [Y], by doing [Z]).
"""
