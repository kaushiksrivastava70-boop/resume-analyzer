"""Resume Analyzer Streamlit Web Application.

A modern, production-grade UI for parsing, evaluating, and optimizing resumes
with ATS scoring, skill detection, job description matching, and downloadable reports.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import plotly.graph_objects as go
import streamlit as st

from analyzer import (
    ResumeAnalysis,
    ResumeExtractionError,
    analyze_resume,
    extract_text,
    generate_pdf_report,
    generate_text_report,
)
from recruiter_mode import render_recruiter_mode


# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Resume Analyzer & ATS Audit",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- CUSTOM CSS INJECTION ---
def inject_custom_css():
    st.markdown(
        """
        <style>
        /* Import clean Inter font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* Hero Banner Styling */
        .hero-container {
            background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 50%, #2563EB 100%);
            border-radius: 16px;
            padding: 2.2rem 2.5rem;
            color: white;
            margin-bottom: 2rem;
            box-shadow: 0 10px 25px -5px rgba(79, 70, 229, 0.25);
        }
        .hero-badge {
            display: inline-block;
            background: rgba(255, 255, 255, 0.2);
            backdrop-filter: blur(8px);
            padding: 4px 14px;
            border-radius: 9999px;
            font-size: 0.82rem;
            font-weight: 600;
            letter-spacing: 0.5px;
            margin-bottom: 0.8rem;
            text-transform: uppercase;
        }
        .hero-title {
            font-size: 2.3rem;
            font-weight: 800;
            line-height: 1.2;
            margin-bottom: 0.5rem;
        }
        .hero-subtitle {
            font-size: 1.05rem;
            opacity: 0.92;
            max-width: 700px;
            line-height: 1.5;
            margin-bottom: 0;
        }

        /* 3-Step Process Flow */
        .step-container {
            display: flex;
            gap: 1rem;
            margin-bottom: 2rem;
            flex-wrap: wrap;
        }
        .step-card {
            flex: 1;
            min-width: 220px;
            background: white;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 1rem 1.25rem;
            display: flex;
            align-items: center;
            gap: 0.85rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        .step-num {
            width: 32px;
            height: 32px;
            border-radius: 50%;
            background: #EEF2FF;
            color: #4F46E5;
            font-weight: 700;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.9rem;
        }
        .step-text strong {
            display: block;
            font-size: 0.92rem;
            color: #1E293B;
        }
        .step-text span {
            font-size: 0.8rem;
            color: #64748B;
        }

        /* Metric Cards */
        .stat-card {
            background: white;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 1.25rem;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.03);
        }
        .stat-value {
            font-size: 1.75rem;
            font-weight: 700;
            color: #1E293B;
            margin-bottom: 0.2rem;
        }
        .stat-label {
            font-size: 0.82rem;
            color: #64748B;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        /* Skill Badges & Chips */
        .skill-badge {
            display: inline-block;
            background: #EEF2FF;
            color: #4338CA;
            border: 1px solid #C7D2FE;
            padding: 5px 12px;
            border-radius: 9999px;
            font-size: 0.84rem;
            font-weight: 500;
            margin: 3px 4px;
        }
        .skill-badge-green {
            background: #ECFDF5;
            color: #065F46;
            border: 1px solid #A7F3D0;
        }
        .skill-badge-red {
            background: #FEF2F2;
            color: #991B1B;
            border: 1px solid #FECACA;
        }
        .skill-badge-amber {
            background: #FFFBEB;
            color: #92400E;
            border: 1px solid #FDE68A;
        }

        /* Priority Tags */
        .priority-high {
            background: #FEE2E2;
            color: #DC2626;
            font-weight: 700;
            padding: 2px 8px;
            border-radius: 6px;
            font-size: 0.75rem;
        }
        .priority-med {
            background: #FEF3C7;
            color: #D97706;
            font-weight: 700;
            padding: 2px 8px;
            border-radius: 6px;
            font-size: 0.75rem;
        }
        .priority-low {
            background: #DBEAFE;
            color: #2563EB;
            font-weight: 700;
            padding: 2px 8px;
            border-radius: 6px;
            font-size: 0.75rem;
        }

        /* Sidebar Styling */
        .sidebar-section {
            background: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            padding: 1rem;
            margin-bottom: 1rem;
            font-size: 0.88rem;
        }

        /* Privacy note */
        .privacy-box {
            background: #F0FDF4;
            border: 1px solid #BBF7D0;
            color: #166534;
            border-radius: 10px;
            padding: 0.85rem 1rem;
            font-size: 0.82rem;
            margin-top: 1rem;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# --- CHART HELPERS ---
def create_gauge_chart(score: float) -> go.Figure:
    """Create a circular gauge chart for overall score with modern color thresholds."""
    if score >= 75:
        bar_color = "#10B981"  # Emerald green
    elif score >= 50:
        bar_color = "#F59E0B"  # Amber
    else:
        bar_color = "#EF4444"  # Red

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "/100", "font": {"size": 42, "color": "#1E293B", "family": "Inter"}},
            title={"text": "Overall ATS Score", "font": {"size": 18, "color": "#64748B", "family": "Inter"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#CBD5E1"},
                "bar": {"color": bar_color, "thickness": 0.3},
                "bgcolor": "#F1F5F9",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 50], "color": "#FEE2E2"},
                    {"range": [50, 75], "color": "#FEF3C7"},
                    {"range": [75, 100], "color": "#D1FAE5"},
                ],
                "threshold": {
                    "line": {"color": "#1E293B", "width": 3},
                    "thickness": 0.8,
                    "value": score,
                },
            },
        )
    )
    fig.update_layout(
        height=280,
        margin=dict(l=25, r=25, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter"},
    )
    return fig


def create_radar_chart(category_scores: dict[str, float]) -> go.Figure:
    """Create a radar chart showing breakdown across evaluated resume dimensions."""
    # Maximum points possible per category for standardizing to 100%
    max_scores = {
        "Contact Info": 15.0,
        "Sections": 20.0,
        "Skills": 25.0,
        "Impact & Metrics": 20.0,
        "Readability & Format": 20.0,
    }

    categories = list(category_scores.keys())
    normalized_values = [
        round((category_scores[cat] / max_scores.get(cat, 20.0)) * 100.0, 1)
        for cat in categories
    ]

    # Close the radar polygon
    categories_closed = categories + [categories[0]]
    values_closed = normalized_values + [normalized_values[0]]

    fig = go.Figure(
        data=go.Scatterpolar(
            r=values_closed,
            theta=categories_closed,
            fill="toself",
            fillcolor="rgba(79, 70, 229, 0.2)",
            line=dict(color="#4F46E5", width=2),
            marker=dict(color="#4338CA", size=6),
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=10, color="#64748B")),
            angularaxis=dict(tickfont=dict(size=11, color="#1E293B", family="Inter")),
        ),
        height=280,
        margin=dict(l=40, r=40, t=25, b=25),
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig


# --- SAMPLE DATA LOADER ---
def load_sample_data() -> tuple[str, str]:
    """Load sample resume text and companion job description."""
    sample_path = Path(__file__).parent / "sample_resume.txt"
    if sample_path.exists():
        resume_text = sample_path.read_text(encoding="utf-8")
    else:
        resume_text = "Sample resume content not found."

    sample_jd = """
Senior Full-Stack Engineer

About the Role:
We are looking for a Senior Full-Stack Software Engineer with 4+ years of hands-on experience building high-scale cloud web systems. You will lead architecture for core web services and drive engineering excellence.

Requirements & Qualifications:
- 4+ years software engineering experience with Python, TypeScript, and React.
- Strong proficiency in modern API frameworks (FastAPI, Next.js, GraphQL).
- Production expertise with AWS, Docker, Kubernetes, and Terraform.
- Solid experience with relational databases (PostgreSQL) and in-memory caches (Redis).
- Familiarity with Golang, Kafka, and PyTorch is a strong plus.
- Proven track record of optimizing system performance and mentoring teammates.
- Excellent communication and cross-functional leadership skills in an Agile environment.
"""
    return resume_text, sample_jd.strip()


# --- MAIN APPLICATION ENTRY ---
def main():
    inject_custom_css()

    # --- SIDEBAR MODE SWITCHER ---
    with st.sidebar:
        st.markdown("### 🎛️ Select App Mode")
        app_mode = st.radio(
            "Select Interface:",
            ["👤 Candidate Mode (Single Resume)", "🏢 Recruiter Mode (Batch Screening)"],
            index=0,
            key="app_mode_selection",
            label_visibility="collapsed",
        )
        st.markdown("---")

    if app_mode.startswith("🏢"):
        with st.sidebar:
            st.markdown("### 📋 Recruiter Instructions")
            st.markdown(
                """
                <div class="sidebar-section">
                    <strong>Batch Screening Steps:</strong>
                    <ol style="margin-top: 6px; padding-left: 18px; margin-bottom: 0;">
                        <li>Review or customize target Job Description.</li>
                        <li>Upload a <code>.zip</code> file of resumes or click <b>Load Demo Batch</b>.</li>
                        <li>Adjust the minimum passing score slider.</li>
                        <li>Explore rankings, analytics, and candidate dossiers.</li>
                        <li>Export leaderboard as CSV or Excel.</li>
                    </ol>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div class="privacy-box">
                    🛡️ <div><strong>100% In-Memory Processing</strong><br/>Resumes are unzipped and evaluated directly in memory without persisting to server storage.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("---")
            st.caption("Recruiter Screening Engine • Built with Streamlit & Plotly")

        render_recruiter_mode()
        return

    # --- CANDIDATE MODE SIDEBAR ---
    with st.sidebar:
        st.markdown("### 📋 Navigation & About")

        st.markdown(
            """
            <div class="sidebar-section">
                <strong>How it works:</strong>
                <ol style="margin-top: 6px; padding-left: 18px; margin-bottom: 0;">
                    <li>Upload your resume (PDF or DOCX).</li>
                    <li>Optionally paste the target Job Description.</li>
                    <li>Get an instant ATS audit score and tailored suggestions.</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### 💡 Pro ATS Tips")
        with st.expander("Top 5 ATS Optimization Rules", expanded=False):
            st.markdown(
                """
                - **Use standard section headers** (Experience, Education, Skills, Projects).
                - **Avoid complex multi-column layouts** or tables that confuse automated parsers.
                - **Include hard metrics** (%, $, time saved, user scale).
                - **Mirror key terms** from the job description naturally.
                - **Keep it to 1–2 pages** (400–900 words).
                """
            )

        st.markdown(
            """
            <div class="privacy-box">
                🛡️ <div><strong>Privacy Guaranteed</strong><br/>Your resume is analyzed completely in-memory and never saved to any database or server disk.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.caption("Resume Analyzer v1.0 • Built with Python & Streamlit")

    # --- HERO SECTION ---
    st.markdown(
        """
        <div class="hero-container">
            <div class="hero-badge">⚡ Next-Gen ATS Resume Optimizer</div>
            <div class="hero-title">Resume Analyzer & Job Matcher</div>
            <p class="hero-subtitle">
                Supercharge your resume with real-time scoring, comprehensive technical skill extraction,
                quantified impact detection, and target job description gap analysis.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- 3-STEP WORKFLOW INDICATOR ---
    st.markdown(
        """
        <div class="step-container">
            <div class="step-card">
                <div class="step-num">1</div>
                <div class="step-text">
                    <strong>Upload Resume</strong>
                    <span>PDF or DOCX format</span>
                </div>
            </div>
            <div class="step-card">
                <div class="step-num">2</div>
                <div class="step-text">
                    <strong>Instant Analysis</strong>
                    <span>Scoring & gap analysis</span>
                </div>
            </div>
            <div class="step-card">
                <div class="step-num">3</div>
                <div class="step-text">
                    <strong>Actionable Improvements</strong>
                    <span>Prioritized recommendations</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- QUICK SAMPLE BUTTON ROW ---
    col_sample_btn, col_clear_btn = st.columns([3, 1])
    with col_sample_btn:
        if st.button("✨ Try Sample Resume (Full-Stack Engineer)", type="primary"):
            st.session_state["use_sample"] = True
    with col_clear_btn:
        if st.button("🔄 Reset"):
            st.session_state["use_sample"] = False
            st.session_state.pop("uploaded_file", None)
            st.session_state.pop("jd_text", None)
            st.rerun()

    use_sample = st.session_state.get("use_sample", False)
    sample_resume_content, sample_jd_content = load_sample_data()

    # --- INPUT SECTION ---
    st.markdown("### 📥 1. Provide Resume & Target Job")
    col_upload, col_jd = st.columns([1, 1], gap="large")

    resume_text = ""
    file_bytes = None
    filename = ""

    with col_upload:
        st.markdown("**Upload Your Resume**")
        uploaded_file = st.file_uploader(
            "Drop your PDF or DOCX file here",
            type=["pdf", "docx"],
            help="Files are processed in memory and never stored.",
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            file_bytes = uploaded_file.read()
            filename = uploaded_file.name
            try:
                resume_text = extract_text(file_bytes, filename)
                st.success(f"Loaded `{filename}` ({len(resume_text.split())} words)")
            except ResumeExtractionError as e:
                st.error(f"⚠️ {str(e)}")
                resume_text = ""
        elif use_sample:
            resume_text = sample_resume_content
            filename = "sample_software_engineer_resume.txt"
            st.info("Loaded pre-built sample resume: **Alex Morgan (Senior Full-Stack Engineer)**")

    with col_jd:
        st.markdown("**Target Job Description (Optional)**")
        default_jd = sample_jd_content if use_sample else ""
        jd_input = st.text_area(
            "Paste target job description to run skill match & keyword gap analysis:",
            value=default_jd,
            height=160,
            placeholder="Paste job description requirements and duties here...",
            label_visibility="collapsed",
        )

    # If no resume provided, prompt user
    if not resume_text:
        st.info("👆 Upload a resume file (.pdf or .docx) or click **'Try Sample Resume'** to get started.")
        return

    # --- ANALYSIS ENGINE EXECUTION ---
    with st.spinner("Analyzing resume content, parsing skills, and calculating metrics..."):
        try:
            analysis: ResumeAnalysis = analyze_resume(resume_text, jd_input)
        except ResumeExtractionError as e:
            st.error(f"Extraction error: {str(e)}")
            return
        except Exception as e:
            st.error(f"Unexpected error analyzing resume: {str(e)}")
            return

    # --- DOWNLOAD ACTION BAR ---
    st.markdown("---")
    col_score_banner, col_dl_pdf, col_dl_txt = st.columns([2, 1, 1], gap="medium")

    with col_score_banner:
        score_color = "#10B981" if analysis.total_score >= 75 else ("#F59E0B" if analysis.total_score >= 50 else "#EF4444")
        st.markdown(
            f"""
            <div style="display:flex; align-items:center; gap:12px; padding: 6px 0;">
                <span style="font-size:1.8rem; font-weight:800; color:{score_color};">{analysis.total_score}/100</span>
                <span style="color:#64748B; font-size:0.95rem;">Overall Evaluation Score</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_dl_pdf:
        pdf_bytes = generate_pdf_report(analysis)
        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_bytes,
            file_name="resume_audit_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    with col_dl_txt:
        txt_report = generate_text_report(analysis)
        st.download_button(
            label="📝 Download Text Report",
            data=txt_report,
            file_name="resume_audit_report.txt",
            mime="text/plain",
            use_container_width=True,
        )

    # --- TABS LAYOUT ---
    tab_overview, tab_skills, tab_jd, tab_suggestions = st.tabs([
        "📊 Overview",
        f"🛠️ Skills ({len(analysis.all_skills)})",
        "🎯 Job Description Match",
        f"💡 Improvement Suggestions ({len(analysis.suggestions)})",
    ])

    # =========================================================================
    # TAB 1: OVERVIEW
    # =========================================================================
    with tab_overview:
        col_gauge, col_radar = st.columns([1, 1])
        with col_gauge:
            st.plotly_chart(create_gauge_chart(analysis.total_score), use_container_width=True)
        with col_radar:
            st.plotly_chart(create_radar_chart(analysis.category_scores), use_container_width=True)

        st.markdown("#### 📌 Key Metrics")
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-value">{analysis.word_count}</div>
                    <div class="stat-label">Word Count</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_m2:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-value">{len(analysis.all_skills)}</div>
                    <div class="stat-label">Skills Detected</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_m3:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-value">{len(analysis.quantified_metrics)}</div>
                    <div class="stat-label">Impact Metrics</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_m4:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-value">{analysis.readability_score}</div>
                    <div class="stat-label">Readability Score</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br/>", unsafe_allow_html=True)
        col_contact_card, col_sections_card = st.columns([1, 1], gap="medium")

        with col_contact_card:
            st.markdown("#### 📇 Contact & Online Presence")
            c = analysis.contact_info
            st.markdown(f"- **Email:** {f'`{c.email}`' if c.email else '❌ Missing'}")
            st.markdown(f"- **Phone:** {f'`{c.phone}`' if c.phone else '❌ Missing'}")
            st.markdown(f"- **LinkedIn:** {f'`{c.linkedin}`' if c.linkedin else '❌ Missing'}")
            st.markdown(f"- **GitHub / Portfolio:** {f'`{c.github or c.portfolio}`' if (c.github or c.portfolio) else '❌ Missing'}")

        with col_sections_card:
            st.markdown("#### 📑 Standard Sections Audit")
            for sec, present in analysis.sections_detected.items():
                icon = "✅" if present else "⚠️"
                status_text = "Detected" if present else "Missing or unlabelled"
                st.markdown(f"- {icon} **{sec}**: {status_text}")

    # =========================================================================
    # TAB 2: SKILLS BREAKDOWN
    # =========================================================================
    with tab_skills:
        st.markdown(f"### Detected Skills Across Categories ({len(analysis.all_skills)})")
        st.caption("Skills are automatically extracted and cross-referenced with modern engineering taxonomies.")

        if not analysis.all_skills:
            st.warning("No recognized technical or soft skills were detected in the resume text.")
        else:
            for category, skills in analysis.skills_by_category.items():
                if skills:
                    st.markdown(f"#### {category} ({len(skills)})")
                    badges_html = " ".join([f'<span class="skill-badge">{s}</span>' for s in skills])
                    st.markdown(badges_html, unsafe_allow_html=True)
                    st.markdown("<br/>", unsafe_allow_html=True)

            empty_categories = [cat for cat, sk in analysis.skills_by_category.items() if len(sk) == 0]
            if empty_categories:
                with st.expander("ℹ️ Unrepresented Skill Categories in This Resume"):
                    st.write(
                        "Consider whether adding skills in these areas aligns with your experience: "
                        + ", ".join(empty_categories)
                    )

    # =========================================================================
    # TAB 3: JOB DESCRIPTION MATCH
    # =========================================================================
    with tab_jd:
        if not analysis.jd_match:
            st.info(
                "💡 **No Job Description Provided**: Paste a target job description in the text box above to "
                "calculate keyword alignment, identify missing requirements, and compute your match percentage."
            )
        else:
            jdm = analysis.jd_match
            st.markdown("### 🎯 Job Description Compatibility")

            col_jd_stat1, col_jd_stat2, col_jd_stat3 = st.columns(3)
            with col_jd_stat1:
                st.metric("Skill Match Rate", f"{jdm.match_percentage}%")
            with col_jd_stat2:
                st.metric("Matched Skills", f"{len(jdm.matched_skills)}")
            with col_jd_stat3:
                st.metric("Missing Skills", f"{len(jdm.missing_skills)}")

            st.progress(jdm.match_percentage / 100.0)

            col_matched, col_missing = st.columns(2, gap="medium")
            with col_matched:
                st.markdown(f"#### ✅ Matched Skills ({len(jdm.matched_skills)})")
                if jdm.matched_skills:
                    matched_html = " ".join([f'<span class="skill-badge skill-badge-green">✓ {s}</span>' for s in jdm.matched_skills])
                    st.markdown(matched_html, unsafe_allow_html=True)
                else:
                    st.caption("No matching skills found between resume and job description.")

            with col_missing:
                st.markdown(f"#### ⚠️ Missing Required Skills ({len(jdm.missing_skills)})")
                if jdm.missing_skills:
                    missing_html = " ".join([f'<span class="skill-badge skill-badge-red">✗ {s}</span>' for s in jdm.missing_skills])
                    st.markdown(missing_html, unsafe_allow_html=True)
                else:
                    st.success("Great job! All technical skills detected in the job description appear in your resume.")

            if jdm.keyword_gaps:
                st.markdown("<br/>", unsafe_allow_html=True)
                st.markdown("#### 🔍 Additional Keyword Gaps")
                st.caption("Frequent terminology found in the Job Description that does not appear in your resume:")
                kw_html = " ".join([f'<span class="skill-badge skill-badge-amber">{kw}</span>' for kw in jdm.keyword_gaps])
                st.markdown(kw_html, unsafe_allow_html=True)

    # =========================================================================
    # TAB 4: SUGGESTIONS
    # =========================================================================
    with tab_suggestions:
        st.markdown(f"### 💡 Actionable Improvement Recommendations ({len(analysis.suggestions)})")
        st.caption("Prioritized steps to improve ATS scan performance and increase recruiter callback rates.")

        if not analysis.suggestions:
            st.success("🎉 Outstanding! Your resume adheres to all recommended ATS and structural standards.")
        else:
            for s in analysis.suggestions:
                badge_class = "priority-high" if s.priority == "High" else ("priority-med" if s.priority == "Medium" else "priority-low")
                with st.expander(f"{s.icon} {s.title} ({s.priority} Priority)", expanded=(s.priority == "High")):
                    st.markdown(f'<span class="{badge_class}">{s.priority.upper()} PRIORITY</span> &nbsp; <b>Category:</b> {s.category}', unsafe_allow_html=True)
                    st.markdown(f"<p style='margin-top:8px; font-size:0.95rem; color:#334155;'>{s.description}</p>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
