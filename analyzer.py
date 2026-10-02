"""Resume Analyzer Core Engine.

Provides text extraction, entity recognition, contact extraction,
section parsing, skill taxonomy matching, impact metric evaluation,
readability scoring, job description comparison, and report generation.
"""

from __future__ import annotations

import io
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import pypdf
from docx import Document
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib import colors

from skills_db import ACTION_VERBS, SECTION_KEYWORDS, SKILLS_TAXONOMY


@dataclass
class ContactInfo:
    email: Optional[str] = None
    phone: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None
    portfolio: Optional[str] = None

    def completeness_score(self) -> float:
        """Returns score out of 15."""
        score = 0.0
        if self.email:
            score += 4.0
        if self.phone:
            score += 4.0
        if self.linkedin:
            score += 3.5
        if self.github or self.portfolio:
            score += 3.5
        return min(15.0, score)


@dataclass
class Suggestion:
    title: str
    description: str
    priority: str  # "High", "Medium", "Low"
    category: str
    icon: str = "💡"


@dataclass
class JDMatchResult:
    match_percentage: float
    matched_skills: List[str]
    missing_skills: List[str]
    jd_skills_count: int
    keyword_gaps: List[str]


@dataclass
class ResumeAnalysis:
    raw_text: str
    word_count: int
    sentence_count: int
    contact_info: ContactInfo
    sections_detected: Dict[str, bool]
    skills_by_category: Dict[str, List[str]]
    all_skills: List[str]
    action_verbs: List[str]
    quantified_metrics: List[str]
    readability_score: float
    category_scores: Dict[str, float]
    total_score: float
    suggestions: List[Suggestion] = field(default_factory=list)
    jd_match: Optional[JDMatchResult] = None


class ResumeExtractionError(Exception):
    """Raised when text cannot be extracted from a resume file."""
    pass


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text content from PDF file bytes.

    Raises:
        ResumeExtractionError: If the PDF is empty, password-protected,
        corrupted, or appears to be a scanned image.
    """
    if not file_bytes:
        raise ResumeExtractionError("The uploaded PDF file is empty (0 bytes).")

    try:
        pdf_file = io.BytesIO(file_bytes)
        reader = pypdf.PdfReader(pdf_file)

        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                raise ResumeExtractionError(
                    "This PDF is password-protected or encrypted. "
                    "Please upload an unprotected version."
                )

        if len(reader.pages) == 0:
            raise ResumeExtractionError("The PDF document contains no pages.")

        extracted_text: List[str] = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            extracted_text.append(text)

        full_text = "\n".join(extracted_text).strip()

        # Check if text is suspiciously short (scanned PDF)
        if len(full_text) < 40 and len(reader.pages) >= 1:
            raise ResumeExtractionError(
                "Very little or no selectable text detected. Your PDF might be a "
                "scanned image or flattened graphic. Please use a text-based PDF or DOCX file."
            )

        return full_text
    except ResumeExtractionError:
        raise
    except Exception as e:
        raise ResumeExtractionError(f"Failed to read PDF document: {str(e)}")


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract text content from DOCX file bytes."""
    if not file_bytes:
        raise ResumeExtractionError("The uploaded DOCX file is empty (0 bytes).")

    try:
        docx_file = io.BytesIO(file_bytes)
        doc = Document(docx_file)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]

        # Also extract table text
        for table in doc.tables:
            for row in table.rows:
                row_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if row_text:
                    paragraphs.append(" | ".join(row_text))

        full_text = "\n".join(paragraphs).strip()
        if not full_text:
            raise ResumeExtractionError(
                "The DOCX document contains no readable text content."
            )
        return full_text
    except ResumeExtractionError:
        raise
    except Exception as e:
        raise ResumeExtractionError(f"Failed to read DOCX document: {str(e)}")


def extract_text(file_bytes: bytes, filename: str) -> str:
    """Extract text from supported file types (PDF, DOCX, TXT)."""
    filename_lower = filename.lower()
    if filename_lower.endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)
    elif filename_lower.endswith(".docx"):
        return extract_text_from_docx(file_bytes)
    elif filename_lower.endswith(".txt"):
        try:
            return file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return file_bytes.decode("latin-1", errors="ignore")
    else:
        raise ResumeExtractionError(
            f"Unsupported file format '{filename}'. Please upload a PDF (.pdf) or Word document (.docx)."
        )


def extract_contact_info(text: str) -> ContactInfo:
    """Extract email, phone, LinkedIn, GitHub, and portfolio links from resume text."""
    # Email regex
    email_pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
    emails = re.findall(email_pattern, text)
    email = emails[0] if emails else None

    # Phone regex supporting international, US standard, dashed, and parenthesized formats
    phone_pattern = r"(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{3,4}\b"
    phones = re.findall(phone_pattern, text)
    valid_phone = None
    for p in phones:
        cleaned = re.sub(r"[^\d]", "", p)
        if 7 <= len(cleaned) <= 15:
            valid_phone = p.strip()
            break

    # LinkedIn
    linkedin_match = re.search(
        r"(?:https?://)?(?:www\.)?linkedin\.com/(?:in|profile)/[a-zA-Z0-9_\-]+",
        text,
        re.IGNORECASE
    )
    linkedin = linkedin_match.group(0) if linkedin_match else None

    # GitHub
    github_match = re.search(
        r"(?:https?://)?(?:www\.)?github\.com/[a-zA-Z0-9_\-]+",
        text,
        re.IGNORECASE
    )
    github = github_match.group(0) if github_match else None

    # General portfolio / personal link
    portfolio = None
    if not (linkedin and github):
        portfolio_match = re.search(
            r"(?:https?://)?(?:www\.)?[a-zA-Z0-9_\-]+\.(?:dev|io|me|tech|app|com|org)(?:/[^\s]*)?",
            text,
            re.IGNORECASE
        )
        if portfolio_match:
            candidate = portfolio_match.group(0)
            if "linkedin.com" not in candidate and "github.com" not in candidate:
                portfolio = candidate

    return ContactInfo(
        email=email,
        phone=valid_phone,
        linkedin=linkedin,
        github=github,
        portfolio=portfolio
    )


def extract_sections(text: str) -> Dict[str, bool]:
    """Detect whether standard resume sections are present."""
    text_lower = text.lower()
    sections_found: Dict[str, bool] = {}

    for section, keywords in SECTION_KEYWORDS.items():
        found = False
        for kw in keywords:
            # Match keyword as whole word or section title pattern
            pattern = rf"(?:^|\n|\r)[ \t]*{re.escape(kw)}[ \t]*(?::|\n|\r|$|-)"
            if re.search(pattern, text_lower, re.MULTILINE):
                found = True
                break
            # Fallback simple search if boundary match failed but word is clearly isolated
            if re.search(rf"\b{re.escape(kw)}\b", text_lower):
                found = True
                break
        sections_found[section] = found

    return sections_found


def _build_skill_pattern(skill: str) -> re.Pattern:
    """Build a regex pattern for a skill accounting for symbols and word boundaries."""
    # For skills with symbols like C++, C#, .NET, Node.js
    escaped = re.escape(skill)
    # If starting/ending with word characters, use \b, otherwise use character boundaries
    prefix = r"\b" if re.match(r"^\w", skill) else r"(?:^|\s|[(\[,])"
    suffix = r"\b" if re.match(r".*\w$", skill) else r"(?:$|\s|[)\],.:;])"
    return re.compile(f"{prefix}{escaped}{suffix}", re.IGNORECASE)


def extract_skills(text: str) -> Tuple[Dict[str, List[str]], List[str]]:
    """Extract categorized skills and distinct list of all found skills."""
    skills_by_category: Dict[str, List[str]] = {}
    all_found: Set[str] = set()

    for category, skills_list in SKILLS_TAXONOMY.items():
        category_matches: List[str] = []
        for skill in skills_list:
            pattern = _build_skill_pattern(skill)
            if pattern.search(text):
                # Avoid duplicates like 'React' and 'React.js' showing redundantly if both matched
                category_matches.append(skill)
                all_found.add(skill)
        skills_by_category[category] = sorted(category_matches)

    return skills_by_category, sorted(list(all_found))


def extract_action_verbs(text: str) -> List[str]:
    """Extract strong impact action verbs used in resume."""
    found: Set[str] = set()
    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    for w in words:
        if w in ACTION_VERBS:
            found.add(w)
    return sorted(list(found))


def extract_quantified_metrics(text: str) -> List[str]:
    """Detect quantified metrics (percentages, dollar amounts, large numbers, performance gains)."""
    metric_patterns = [
        r"\b\d+(?:\.\d+)?%",  # e.g., 42%, 99.99%
        r"\$\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:k|m|b|million|billion|k/yr))?",  # $45,000/yr, $8.5M
        r"\b\d+(?:\.\d+)?\s*(?:x|times)\b",  # 10x, 2 times
        r"\b\d+(?:,\d{3})*(?:\.\d+)?[kKmMbB]?\+?\s*(?:users|clients|teams|engineers|repositories|requests|downloads|stars|deliveries|ms|seconds|minutes|hours)\b",  # 10M+ users, 18ms
        r"\b\d+[kKmMbB]\+?\b",  # 10M+, 500k
    ]
    matches: List[str] = []
    for pattern in metric_patterns:
        found = re.findall(pattern, text, re.IGNORECASE)
        matches.extend(found)

    # Return unique matches preserved in order of appearance
    seen = set()
    unique_matches = []
    for m in matches:
        cleaned = m.strip()
        if cleaned.lower() not in seen:
            seen.add(cleaned.lower())
            unique_matches.append(cleaned)
    return unique_matches


def calculate_readability(text: str) -> Tuple[int, int, float]:
    """Calculate word count, sentence count, and estimated Flesch Reading Ease score."""
    clean_text = text.strip()
    if not clean_text:
        return 0, 0, 0.0

    words = re.findall(r"\b[a-zA-Z0-9'-]+\b", clean_text)
    word_count = len(words)
    if word_count == 0:
        return 0, 0, 0.0

    # Sentences split by punctuation or newlines
    sentences = [s.strip() for s in re.split(r"[.!?\n]+", clean_text) if len(s.strip()) > 3]
    sentence_count = max(len(sentences), 1)

    # Count syllables roughly
    def count_syllables(word: str) -> int:
        w = word.lower()
        w = re.sub(r"(?:[^laeiouy]|ed|es|e)$", "", w)
        w = re.sub(r"^y", "", w)
        syl = len(re.findall(r"[aeiouy]{1,2}", w))
        return max(syl, 1)

    total_syllables = sum(count_syllables(w) for w in words)

    # Flesch Reading Ease formula: 206.835 - 1.015 * (total words / total sentences) - 84.6 * (total syllables / total words)
    asl = word_count / sentence_count
    asw = total_syllables / word_count
    fre = 206.835 - (1.015 * asl) - (84.6 * asw)

    # Normalization for resumes (typically denser than literature: standard good resume sits between 45 and 75)
    normalized_score = max(0.0, min(100.0, fre))
    return word_count, sentence_count, round(normalized_score, 1)


def score_resume(
    contact_info: ContactInfo,
    sections: Dict[str, bool],
    all_skills: List[str],
    action_verbs: List[str],
    quantified_metrics: List[str],
    word_count: int,
    readability: float
) -> Tuple[Dict[str, float], float]:
    """Compute score breakdown (out of 100) and overall weighted score.

    Breakdown:
    1. Contact Info & Links: 15 points
    2. Sections Present: 20 points
    3. Skills & Breadth: 25 points
    4. Action Verbs & Metrics: 20 points
    5. Length & Readability: 20 points
    """
    scores: Dict[str, float] = {}

    # 1. Contact Info (15 pts)
    scores["Contact Info"] = round(contact_info.completeness_score(), 1)

    # 2. Sections (20 pts)
    section_weights = {
        "Work Experience": 5.0,
        "Skills": 4.0,
        "Education": 4.0,
        "Projects": 4.0,
        "Summary / Objective": 3.0,
    }
    sec_score = 0.0
    for sec, wt in section_weights.items():
        if sections.get(sec, False):
            sec_score += wt
    # Bonus for Certifications if present (up to 20 max)
    if sections.get("Certifications", False):
        sec_score = min(20.0, sec_score + 1.0)
    scores["Sections"] = round(min(20.0, sec_score), 1)

    # 3. Skills (25 pts)
    # Target: 12+ skills yields full points
    skill_count = len(all_skills)
    if skill_count >= 14:
        skill_score = 25.0
    elif skill_count >= 10:
        skill_score = 22.0
    elif skill_count >= 6:
        skill_score = 17.0
    elif skill_count >= 3:
        skill_score = 10.0
    else:
        skill_score = skill_count * 2.5
    scores["Skills"] = round(min(25.0, skill_score), 1)

    # 4. Action Verbs & Impact Metrics (20 pts)
    # Action verbs (10 pts): 8+ verbs = 10 pts
    verb_score = min(10.0, len(action_verbs) * 1.25)
    # Metrics (10 pts): 5+ quantified metrics = 10 pts
    metric_score = min(10.0, len(quantified_metrics) * 2.0)
    scores["Impact & Metrics"] = round(min(20.0, verb_score + metric_score), 1)

    # 5. Length & Readability (20 pts)
    # Ideal resume length: 300 to 850 words (1-2 pages)
    length_score = 10.0
    if word_count < 150:
        length_score = 3.0
    elif word_count < 250:
        length_score = 6.0
    elif word_count > 1200:
        length_score = 6.0
    elif word_count > 1600:
        length_score = 4.0

    # Readability: 40-75 is ideal for technical documents
    if 35.0 <= readability <= 75.0:
        read_score = 10.0
    elif 25.0 <= readability < 35.0 or 75.0 < readability <= 85.0:
        read_score = 7.5
    else:
        read_score = 5.0
    scores["Readability & Format"] = round(length_score + read_score, 1)

    total = sum(scores.values())
    return scores, round(min(100.0, max(0.0, total)), 1)


def generate_suggestions(
    contact_info: ContactInfo,
    sections: Dict[str, bool],
    skills_by_category: Dict[str, List[str]],
    all_skills: List[str],
    action_verbs: List[str],
    quantified_metrics: List[str],
    word_count: int,
    readability: float,
    jd_match: Optional[JDMatchResult] = None
) -> List[Suggestion]:
    """Generate prioritized, actionable improvement recommendations."""
    suggestions: List[Suggestion] = []

    # High Priority: Contact details
    if not contact_info.email:
        suggestions.append(Suggestion(
            title="Add a professional email address",
            description="Recruiters and automated systems require a direct email to initiate interview invitations.",
            priority="High",
            category="Contact Info",
            icon="✉️"
        ))

    if not contact_info.phone:
        suggestions.append(Suggestion(
            title="Include a contact phone number",
            description="Recruiter phone screens are the primary first step for most tech hiring pipelines.",
            priority="High",
            category="Contact Info",
            icon="📞"
        ))

    # High Priority: Missing key sections
    if not sections.get("Work Experience", False):
        suggestions.append(Suggestion(
            title="Add a dedicated 'Work Experience' section",
            description="Experience is the #1 section recruiters look for. Structure it with clear job titles, companies, dates, and bullet points.",
            priority="High",
            category="Structure",
            icon="💼"
        ))

    if not sections.get("Skills", False):
        suggestions.append(Suggestion(
            title="Add a dedicated 'Technical Skills' section",
            description="Grouping skills into clear categories (Languages, Frameworks, Cloud, Databases) helps ATS scanners parse your qualifications.",
            priority="High",
            category="Structure",
            icon="🛠️"
        ))

    # High Priority: Lack of quantified impact
    if len(quantified_metrics) < 3:
        suggestions.append(Suggestion(
            title="Quantify your achievements with numbers & metrics",
            description=f"Only {len(quantified_metrics)} metrics detected. Use metrics like percentages, latency improvements (e.g. 'reduced by 35%'), revenue generated, or user scale ('served 1M+ requests').",
            priority="High",
            category="Impact",
            icon="📈"
        ))

    # Medium Priority: Links
    if not contact_info.linkedin:
        suggestions.append(Suggestion(
            title="Add your LinkedIn profile URL",
            description="Over 85% of technical recruiters cross-reference LinkedIn profiles during application screening.",
            priority="Medium",
            category="Online Presence",
            icon="🔗"
        ))

    if not contact_info.github and not contact_info.portfolio:
        suggestions.append(Suggestion(
            title="Add a GitHub or technical portfolio link",
            description="Providing links to code repositories or deployed demos gives technical interviewers proof of your hands-on ability.",
            priority="Medium",
            category="Online Presence",
            icon="💻"
        ))

    # Medium Priority: Action verbs
    if len(action_verbs) < 5:
        suggestions.append(Suggestion(
            title="Start bullet points with strong action verbs",
            description=f"Detected {len(action_verbs)} unique action verbs. Replace passive phrasing like 'Responsible for' with impact verbs such as 'Architected', 'Spearheaded', 'Optimized', or 'Automated'.",
            priority="Medium",
            category="Language",
            icon="⚡"
        ))

    # Medium Priority: Skill breadth
    empty_categories = [cat for cat, sk in skills_by_category.items() if len(sk) == 0]
    if empty_categories and "Cloud & DevOps" in empty_categories:
        suggestions.append(Suggestion(
            title="Highlight Cloud or DevOps competencies",
            description="Modern engineering roles frequently expect familiarity with Docker, CI/CD pipelines, or cloud providers (AWS, Azure, GCP).",
            priority="Medium",
            category="Skills",
            icon="☁️"
        ))

    # Low Priority: Word count & structure
    if word_count < 250:
        suggestions.append(Suggestion(
            title="Expand resume content (currently under 250 words)",
            description=f"Your resume is only {word_count} words. Flesh out project descriptions, technologies utilized, and specific business problems solved.",
            priority="Low",
            category="Length",
            icon="📝"
        ))
    elif word_count > 1200:
        suggestions.append(Suggestion(
            title="Condense resume for conciseness (over 1,200 words)",
            description=f"Your resume is {word_count} words. Aim for 400 - 800 words (1 to 2 pages) to ensure recruiters can scan it in under 10 seconds.",
            priority="Low",
            category="Length",
            icon="✂️"
        ))

    if not sections.get("Summary / Objective", False):
        suggestions.append(Suggestion(
            title="Consider adding a 2-3 sentence Professional Summary",
            description="A punchy professional summary at the top highlights your years of experience, core tech stack, and primary career accomplishments immediately.",
            priority="Low",
            category="Structure",
            icon="📋"
        ))

    # JD Match specific recommendations
    if jd_match and jd_match.missing_skills:
        missing_preview = ", ".join(jd_match.missing_skills[:5])
        suggestions.append(Suggestion(
            title=f"Address Job Description skill gaps: {missing_preview}",
            description=f"The target job requires keywords missing from your resume: {', '.join(jd_match.missing_skills[:8])}. If you have experience with these, incorporate them into your bullet points.",
            priority="High",
            category="JD Alignment",
            icon="🎯"
        ))

    # Sort suggestions by priority: High first, then Medium, then Low
    priority_order = {"High": 0, "Medium": 1, "Low": 2}
    suggestions.sort(key=lambda s: priority_order.get(s.priority, 3))
    return suggestions


def match_job_description(resume_text: str, jd_text: str) -> JDMatchResult:
    """Compare resume text against target job description for skills and keyword overlap."""
    if not jd_text or not jd_text.strip():
        return JDMatchResult(
            match_percentage=0.0,
            matched_skills=[],
            missing_skills=[],
            jd_skills_count=0,
            keyword_gaps=[]
        )

    # Extract skills from JD
    _, jd_skills = extract_skills(jd_text)
    _, resume_skills = extract_skills(resume_text)

    resume_skills_set = set(s.lower() for s in resume_skills)

    matched = []
    missing = []
    for skill in jd_skills:
        if skill.lower() in resume_skills_set:
            matched.append(skill)
        else:
            missing.append(skill)

    total_jd_skills = len(jd_skills)
    match_percentage = (len(matched) / total_jd_skills * 100.0) if total_jd_skills > 0 else 0.0

    # Keyword gap analysis (frequent non-stopword tokens in JD not present in resume)
    stopwords = {
        "and", "the", "to", "of", "in", "a", "for", "with", "is", "on", "that",
        "by", "this", "an", "be", "are", "from", "at", "as", "your", "all",
        "have", "new", "more", "will", "our", "you", "we", "or", "us", "about",
        "can", "if", "my", "so", "up", "out", "no", "do", "its", "their", "into",
        "experience", "years", "role", "work", "team", "looking", "candidate", "responsibilities"
    }

    jd_words = re.findall(r"\b[a-zA-Z]{3,}\b", jd_text.lower())
    resume_words_set = set(re.findall(r"\b[a-zA-Z]{3,}\b", resume_text.lower()))

    word_freq: Dict[str, int] = {}
    for w in jd_words:
        if w not in stopwords and w not in resume_words_set:
            word_freq[w] = word_freq.get(w, 0) + 1

    # Top missing keywords by frequency
    sorted_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    keyword_gaps = [w.capitalize() for w, count in sorted_keywords[:15]]

    return JDMatchResult(
        match_percentage=round(match_percentage, 1),
        matched_skills=matched,
        missing_skills=missing,
        jd_skills_count=total_jd_skills,
        keyword_gaps=keyword_gaps
    )


def analyze_resume(resume_text: str, jd_text: Optional[str] = None) -> ResumeAnalysis:
    """Full end-to-end analysis of resume text with optional JD matching."""
    if not resume_text or not resume_text.strip():
        raise ResumeExtractionError("The resume contains no text to analyze.")

    word_count, sentence_count, readability = calculate_readability(resume_text)
    contact_info = extract_contact_info(resume_text)
    sections = extract_sections(resume_text)
    skills_by_category, all_skills = extract_skills(resume_text)
    action_verbs = extract_action_verbs(resume_text)
    quantified_metrics = extract_quantified_metrics(resume_text)

    category_scores, total_score = score_resume(
        contact_info=contact_info,
        sections=sections,
        all_skills=all_skills,
        action_verbs=action_verbs,
        quantified_metrics=quantified_metrics,
        word_count=word_count,
        readability=readability
    )

    jd_match = None
    if jd_text and jd_text.strip():
        jd_match = match_job_description(resume_text, jd_text)

    suggestions = generate_suggestions(
        contact_info=contact_info,
        sections=sections,
        skills_by_category=skills_by_category,
        all_skills=all_skills,
        action_verbs=action_verbs,
        quantified_metrics=quantified_metrics,
        word_count=word_count,
        readability=readability,
        jd_match=jd_match
    )

    return ResumeAnalysis(
        raw_text=resume_text,
        word_count=word_count,
        sentence_count=sentence_count,
        contact_info=contact_info,
        sections_detected=sections,
        skills_by_category=skills_by_category,
        all_skills=all_skills,
        action_verbs=action_verbs,
        quantified_metrics=quantified_metrics,
        readability_score=readability,
        category_scores=category_scores,
        total_score=total_score,
        suggestions=suggestions,
        jd_match=jd_match
    )


def generate_text_report(analysis: ResumeAnalysis) -> str:
    """Generate a clean, structured text report of the analysis."""
    lines = [
        "=" * 60,
        "           RESUME ANALYZER AUDIT REPORT",
        "=" * 60,
        f"Overall Resume Score: {analysis.total_score} / 100",
        f"Total Words: {analysis.word_count} | Sentences: {analysis.sentence_count}",
        f"Readability Score: {analysis.readability_score} / 100",
        "",
        "--- CATEGORY BREAKDOWN ---",
    ]

    for cat, sc in analysis.category_scores.items():
        lines.append(f"  * {cat}: {sc}")

    lines.extend([
        "",
        "--- CONTACT INFORMATION ---",
        f"  * Email: {analysis.contact_info.email or 'MISSING'}",
        f"  * Phone: {analysis.contact_info.phone or 'MISSING'}",
        f"  * LinkedIn: {analysis.contact_info.linkedin or 'MISSING'}",
        f"  * GitHub: {analysis.contact_info.github or 'MISSING'}",
        "",
        "--- DETECTED SECTIONS ---",
    ])

    for sec, present in analysis.sections_detected.items():
        status = "Present [✓]" if present else "Missing [X]"
        lines.append(f"  * {sec}: {status}")

    lines.extend([
        "",
        f"--- DETECTED SKILLS ({len(analysis.all_skills)}) ---"
    ])
    for cat, skills in analysis.skills_by_category.items():
        if skills:
            lines.append(f"  [{cat}]: {', '.join(skills)}")

    lines.extend([
        "",
        f"--- QUANTIFIED IMPACT METRICS ({len(analysis.quantified_metrics)}) ---",
        f"  {', '.join(analysis.quantified_metrics) if analysis.quantified_metrics else 'None detected.'}",
        "",
        f"--- STRONG ACTION VERBS ({len(analysis.action_verbs)}) ---",
        f"  {', '.join(analysis.action_verbs) if analysis.action_verbs else 'None detected.'}",
    ])

    if analysis.jd_match:
        lines.extend([
            "",
            "--- JOB DESCRIPTION MATCH ---",
            f"  Match Percentage: {analysis.jd_match.match_percentage}%",
            f"  Matched Skills: {', '.join(analysis.jd_match.matched_skills) or 'None'}",
            f"  Missing Skills: {', '.join(analysis.jd_match.missing_skills) or 'None'}",
            f"  Keyword Gap: {', '.join(analysis.jd_match.keyword_gaps[:10]) or 'None'}"
        ])

    lines.extend([
        "",
        "--- ACTIONABLE IMPROVEMENTS ---"
    ])
    for i, s in enumerate(analysis.suggestions, 1):
        lines.append(f"{i}. [{s.priority} Priority] {s.title}")
        lines.append(f"   {s.description}")

    lines.append("\nReport generated by Resume Analyzer (Offline & Private).")
    return "\n".join(lines)


def generate_pdf_report(analysis: ResumeAnalysis) -> bytes:
    """Generate a clean, printable PDF report using ReportLab."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#4F46E5"),
        spaceAfter=6
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.HexColor("#64748B"),
        spaceAfter=14
    )
    h2_style = ParagraphStyle(
        "H2",
        parent=styles["Heading2"],
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#1E293B"),
        spaceBefore=10,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#334155")
    )
    badge_style = ParagraphStyle(
        "Badge",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#FFFFFF")
    )

    story = []

    # Title & Header
    story.append(Paragraph("Resume Analysis & ATS Audit Report", title_style))
    story.append(Paragraph(
        f"Generated by Resume Analyzer &bull; Score: <b>{analysis.total_score} / 100</b> &bull; Private & Secure",
        subtitle_style
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#E2E8F0"), spaceAfter=12))

    # Overview Metrics Table
    metrics_data = [
        ["Overall Score", "Readability", "Word Count", "Skills Detected", "Impact Metrics"],
        [
            f"{analysis.total_score} / 100",
            f"{analysis.readability_score} / 100",
            str(analysis.word_count),
            str(len(analysis.all_skills)),
            str(len(analysis.quantified_metrics))
        ]
    ]
    metrics_table = Table(metrics_data, colWidths=[108] * 5)
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor("#1E293B")),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, 1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 12))

    # Category Breakdown
    story.append(Paragraph("Category Score Breakdown", h2_style))
    cat_data = [["Category", "Score"]]
    for cat, sc in analysis.category_scores.items():
        cat_data.append([cat, f"{sc} pts"])
    cat_table = Table(cat_data, colWidths=[270, 270])
    cat_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#EEF2FF")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor("#4F46E5")),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
    ]))
    story.append(cat_table)
    story.append(Spacer(1, 10))

    # Detected Skills
    story.append(Paragraph(f"Detected Skills ({len(analysis.all_skills)})", h2_style))
    for cat, skills in analysis.skills_by_category.items():
        if skills:
            story.append(Paragraph(f"<b>{cat}:</b> {', '.join(skills)}", body_style))
            story.append(Spacer(1, 2))
    story.append(Spacer(1, 10))

    # JD Match (if provided)
    if analysis.jd_match:
        story.append(Paragraph(f"Job Description Alignment (Match: {analysis.jd_match.match_percentage}%)", h2_style))
        story.append(Paragraph(f"<b>Matched Skills ({len(analysis.jd_match.matched_skills)}):</b> {', '.join(analysis.jd_match.matched_skills) or 'None'}", body_style))
        story.append(Paragraph(f"<b>Missing Skills ({len(analysis.jd_match.missing_skills)}):</b> {', '.join(analysis.jd_match.missing_skills) or 'None'}", body_style))
        story.append(Spacer(1, 10))

    # Actionable Improvements
    story.append(Paragraph("Actionable Recommendations", h2_style))
    for s in analysis.suggestions:
        p_color = "#DC2626" if s.priority == "High" else ("#D97706" if s.priority == "Medium" else "#2563EB")
        story.append(Paragraph(
            f"<b><font color='{p_color}'>[{s.priority.upper()}]</font> {s.title}</b><br/>{s.description}",
            body_style
        ))
        story.append(Spacer(1, 4))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
