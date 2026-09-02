import io
import json
import os
import re
from typing import Any, Dict, List

import streamlit as st
from docx import Document
from google import genai
from google.genai import types
from pypdf import PdfReader


APP_TITLE = "Resume ATS Analyzer"
MODEL_NAME = "gemini-3.6-flash"

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📄",
    layout="wide",
)


def get_api_key() -> str:
    """Read the Gemini API key from Streamlit secrets or an environment variable."""
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        key = ""

    return (key or os.getenv("GEMINI_API_KEY", "")).strip()


def extract_pdf_text(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages).strip()


def extract_docx_text(file_bytes: bytes) -> str:
    document = Document(io.BytesIO(file_bytes))
    parts = [p.text for p in document.paragraphs if p.text.strip()]

    # Also read simple table-based resumes.
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                parts.append(" | ".join(cells))

    return "\n".join(parts).strip()


def extract_text(uploaded_file) -> str:
    file_bytes = uploaded_file.getvalue()
    suffix = uploaded_file.name.lower().rsplit(".", 1)[-1]

    if suffix == "pdf":
        return extract_pdf_text(file_bytes)
    if suffix == "docx":
        return extract_docx_text(file_bytes)
    if suffix == "txt":
        return file_bytes.decode("utf-8", errors="replace").strip()

    raise ValueError("Unsupported file type. Please upload PDF, DOCX, or TXT.")


def clean_json_text(text: str) -> str:
    """Remove markdown code fences if Gemini returns them."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def analyze_resume(resume_text: str, job_description: str, api_key: str) -> Dict[str, Any]:
    client = genai.Client(api_key=api_key)

    jd_section = (
        job_description.strip()
        if job_description.strip()
        else "No job description was provided. Evaluate general ATS readiness and resume quality."
    )

    prompt = f"""
You are an expert ATS resume evaluator and technical recruiter.

Analyze the resume below. Return ONLY valid JSON matching the schema at the end.
Do not use markdown. Do not invent facts that are not in the resume.

Important scoring rule:
- "ats_score" must be an integer from 0 to 100.
- If a job description is provided, score keyword alignment and relevance against it.
- If no job description is provided, score general ATS compatibility, clarity, structure,
  measurable achievements, standard section headings, skills, formatting signals visible
  in extracted text, and completeness.
- The score is an estimate, not a claim about any specific ATS vendor.

Evaluate:
1. ATS compatibility and parseability.
2. Keyword alignment.
3. Standard section structure.
4. Skills and role relevance.
5. Achievement-oriented bullet quality.
6. Use of measurable results.
7. Clarity, conciseness, and consistency.
8. Potential ATS risks such as unusual headings, missing contact details,
   excessive symbols, repeated keywords, or information that may parse poorly.

Return:
{{
  "ats_score": 0,
  "score_label": "Needs Improvement",
  "summary": "Short overall assessment.",
  "score_breakdown": {{
    "parseability": 0,
    "keyword_alignment": 0,
    "section_structure": 0,
    "skills_relevance": 0,
    "achievement_quality": 0
  }},
  "strengths": ["...", "..."],
  "improvements": [
    {{
      "priority": "High",
      "issue": "What should change",
      "recommendation": "Specific action",
      "example": "A concise example rewrite when appropriate"
    }}
  ],
  "missing_keywords": ["keyword1", "keyword2"],
  "good_keywords": ["keyword1", "keyword2"],
  "ats_risks": ["risk1", "risk2"],
  "section_feedback": [
    {{
      "section": "Experience",
      "status": "Good",
      "feedback": "Specific feedback"
    }}
  ]
}}

Use 0-100 values for every score in score_breakdown.
Keep lists useful and concise. Do not exceed 8 items in any list.

JOB DESCRIPTION:
{jd_section}

RESUME:
{resume_text[:30000]}
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json",
        ),
    )

    raw = clean_json_text(response.text or "")
    result = json.loads(raw)

    # Basic validation so malformed/unsafe model output doesn't break the UI.
    score = int(result.get("ats_score", 0))
    result["ats_score"] = max(0, min(100, score))

    breakdown = result.get("score_breakdown", {})
    for key, value in list(breakdown.items()):
        try:
            breakdown[key] = max(0, min(100, int(value)))
        except (TypeError, ValueError):
            breakdown[key] = 0

    return result


def score_color(score: int) -> str:
    if score >= 80:
        return "🟢"
    if score >= 60:
        return "🟡"
    return "🔴"


st.title("📄 Resume ATS Analyzer")
st.caption("Upload a resume and get an AI-powered ATS-readiness score and actionable improvements.")

with st.sidebar:
    st.header("Settings")
    st.markdown(
        "This app uses **Gemini 2.5 Flash** to evaluate the extracted resume text."
    )
    st.info(
        "For a stronger ATS estimate, paste the target job description. "
        "Without one, the app evaluates general ATS readiness."
    )
    st.markdown("**Supported files:** PDF, DOCX, TXT")

uploaded_file = st.file_uploader(
    "Upload your resume",
    type=["pdf", "docx", "txt"],
    help="Use a text-based PDF where possible. Scanned/image-only PDFs may not extract correctly.",
)

job_description = st.text_area(
    "Target job description (optional)",
    height=220,
    placeholder="Paste the job description here to compare your resume against the role...",
)

if uploaded_file:
    try:
        resume_text = extract_text(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read this file: {exc}")
        st.stop()

    if not resume_text:
        st.error(
            "No readable text was found. If this is a scanned PDF, convert it to a "
            "text-based PDF or upload a DOCX/TXT version."
        )
        st.stop()

    with st.expander("Preview extracted resume text"):
        st.text(resume_text[:12000])

    api_key = get_api_key()

    if not api_key:
        st.warning(
            "Gemini API key not found. Add `GEMINI_API_KEY` to Streamlit Secrets "
            "or set it as an environment variable."
        )
        st.code('GEMINI_API_KEY = "your-api-key-here"', language="toml")
        st.stop()

    if st.button("🔍 Analyze Resume", type="primary", use_container_width=True):
        with st.spinner("Gemini is analyzing your resume..."):
            try:
                result = analyze_resume(resume_text, job_description, api_key)
            except json.JSONDecodeError:
                st.error("Gemini returned an unexpected response. Please try again.")
                st.stop()
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")
                st.stop()

        score = result["ats_score"]
        st.divider()

        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Estimated ATS Score", f"{score}/100")
            st.subheader(f"{score_color(score)} {result.get('score_label', 'Assessment')}")

        with c2:
            st.write(result.get("summary", "No summary returned."))

        st.subheader("📊 Score Breakdown")
        breakdown = result.get("score_breakdown", {})
        if breakdown:
            cols = st.columns(len(breakdown))
            for col, (name, value) in zip(cols, breakdown.items()):
                col.metric(name.replace("_", " ").title(), f"{value}/100")

        left, right = st.columns(2)

        with left:
            st.subheader("✅ Strengths")
            for item in result.get("strengths", []):
                st.markdown(f"- {item}")

            st.subheader("🔑 Strong Keywords")
            keywords = result.get("good_keywords", [])
            st.write(", ".join(keywords) if keywords else "No specific keywords identified.")

        with right:
            st.subheader("⚠️ ATS Risks")
            risks = result.get("ats_risks", [])
            for item in risks:
                st.markdown(f"- {item}")

            st.subheader("🔎 Missing / Useful Keywords")
            missing = result.get("missing_keywords", [])
            st.write(", ".join(missing) if missing else "No major missing keywords identified.")

        st.subheader("🛠️ Recommended Improvements")
        improvements = result.get("improvements", [])
        if not improvements:
            st.success("No major improvements were returned.")
        else:
            for i, item in enumerate(improvements, 1):
                priority = item.get("priority", "Medium")
                st.markdown(f"**{i}. [{priority}] {item.get('issue', 'Improvement')}**")
                st.write(item.get("recommendation", ""))
                if item.get("example"):
                    st.caption(f"Example: {item['example']}")

        st.subheader("📌 Section-by-Section Feedback")
        for item in result.get("section_feedback", []):
            status = item.get("status", "Review")
            with st.expander(f"{item.get('section', 'Section')} — {status}"):
                st.write(item.get("feedback", ""))

        st.divider()
        st.caption(
            "Note: ATS scores are estimates generated from the resume text and, when provided, "
            "the target job description. Different ATS platforms use different parsing and ranking methods."
        )
else:
    st.info("Upload a PDF, DOCX, or TXT resume to begin.")
