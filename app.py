"""
ATS Resume Checker
Upload a resume and get an ATS score, keyword analysis,
and practical resume improvement suggestions.
"""

import json
import re
import zipfile
import xml.etree.ElementTree as ET
from io import BytesIO

import streamlit as st
from docx import Document
from google import genai
from google.genai import types
from pypdf import PdfReader


# -------------------- CONFIGURATION --------------------

MODEL_OPTIONS = [
    "gemini-3.8-flash",
    "gemini-flash-latest",
    "gemini-3.1-flash-lite",
]

MAX_RESUME_CHARS = 20000
MAX_FILE_MB = 5

SCORE_KEYS = {
    "keywords_match": "Keywords & Skills",
    "formatting": "Formatting & ATS Parsability",
    "structure": "Sections & Structure",
    "impact": "Achievements & Impact",
    "readability": "Clarity & Readability",
}


# -------------------- RESUME TEXT EXTRACTION --------------------

def extract_docx_text(data: bytes) -> str:
    """
    Read text from a DOCX file.
    If Word's image files cause a CRC error, read document.xml directly.
    """

    # Method 1: Try the normal DOCX reader
    try:
        doc = Document(BytesIO(data))
        parts = []

        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                parts.append(paragraph.text.strip())

        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    if cell.text.strip():
                        parts.append(cell.text.strip())

        extracted = "\n".join(parts).strip()

        if extracted:
            return extracted

    except Exception:
        pass

    # Method 2: Extract text directly from the document XML
    # This can work when an embedded image is corrupted.
    try:
        with zipfile.ZipFile(BytesIO(data), "r") as archive:
            xml_data = archive.read("word/document.xml")

        root = ET.fromstring(xml_data)

        namespace = {
            "w": (
                "http://schemas.openxmlformats.org/"
                "wordprocessingml/2006/main"
            )
        }

        paragraphs = []

        for paragraph in root.findall(".//w:p", namespace):
            words = [
                node.text or ""
                for node in paragraph.findall(".//w:t", namespace)
            ]

            line = "".join(words).strip()

            if line:
                paragraphs.append(line)

        extracted = "\n".join(paragraphs).strip()

        if extracted:
            return extracted

        raise ValueError("No readable text was found in this DOCX file.")

    except Exception as exc:
        raise ValueError(
            "The DOCX file could not be read. Open it in Microsoft Word "
            "and save a new copy, or upload a PDF or TXT file instead."
        ) from exc


def extract_text(file_name: str, data: bytes) -> str:
    """Extract text from PDF, DOCX, or TXT files."""

    name = file_name.lower()

    if name.endswith(".pdf"):
        try:
            reader = PdfReader(BytesIO(data))

            if reader.is_encrypted:
                try:
                    reader.decrypt("")
                except Exception as exc:
                    raise ValueError(
                        "This PDF is password-protected. "
                        "Please upload an unlocked PDF."
                    ) from exc

            pages = [
                page.extract_text() or ""
                for page in reader.pages
            ]

            return "\n".join(pages).strip()

        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(
                "Could not read this PDF. Please upload a valid PDF."
            ) from exc

    if name.endswith(".docx"):
        return extract_docx_text(data)

    if name.endswith(".txt"):
        return data.decode("utf-8", errors="ignore").strip()

    raise ValueError(
        "Unsupported file type. Upload a PDF, DOCX, or TXT file."
    )


# -------------------- AI PROMPT --------------------

def build_prompt(resume_text: str, job_description: str) -> str:

    if job_description.strip():
        jd_block = f"JOB DESCRIPTION:\n{job_description.strip()}"
    else:
        jd_block = (
            "JOB DESCRIPTION: Not provided. Evaluate against general "
            "ATS best practices and the likely target role."
        )

    return f"""
You are an expert ATS resume analyst and professional resume coach.

Analyze the resume carefully and return ONLY a valid JSON object.
Do not include Markdown fences or commentary.

Use exactly this structure:

{{
  "overall_score": 0,
  "score_breakdown": {{
    "keywords_match": 0,
    "formatting": 0,
    "structure": 0,
    "impact": 0,
    "readability": 0
  }},
  "summary": "A short, honest assessment.",
  "strengths": ["Strength 1", "Strength 2"],
  "weaknesses": ["Weakness 1", "Weakness 2"],
  "matched_keywords": ["Keyword 1"],
  "missing_keywords": ["Keyword 2"],
  "improvements": [
    {{
      "priority": "High",
      "section": "Experience",
      "issue": "Explain the problem.",
      "suggestion": "Give a specific improvement."
    }}
  ],
  "rewrite_examples": [
    {{
      "original": "Original resume sentence.",
      "improved": "Improved version using only real facts."
    }}
  ]
}}

SCORING RULES:
- All scores must be integers from 0 to 100.
- Be realistic. A score above 90 should be rare.
- Evaluate keyword matching against the job description when provided.
- Check contact details, skills, education, experience, and summary.
- Check ATS readability, formatting, headings, and section structure.
- Check grammar, vague wording, achievements, and measurable results.
- Give 5 to 10 practical improvements, ordered High, Medium, then Low.
- Give 2 to 4 rewrite examples when possible.
- Never invent skills, qualifications, jobs, or achievements.
- Do not recommend adding a keyword as a skill unless it is truthful.
- If a job description is missing, clearly state that keyword matching
  is only a general estimate.
- This is an estimated assessment, not an actual ATS system result.

{jd_block}

RESUME:
{resume_text}
"""


# -------------------- PROCESS AI RESPONSE --------------------

def parse_json_response(raw: str) -> dict:
    """Convert the AI response into a Python dictionary."""

    if not raw:
        raise ValueError("The AI returned an empty response.")

    cleaned = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        raw.strip(),
        flags=re.IGNORECASE,
    )

    try:
        return json.loads(cleaned)

    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")

        if start != -1 and end > start:
            return json.loads(cleaned[start:end + 1])

        raise ValueError("The AI response was not valid JSON.")


def clamp_score(value, default=0) -> int:
    """Keep a score between 0 and 100."""

    try:
        return max(0, min(100, int(round(float(value)))))
    except (TypeError, ValueError):
        return default


def string_list(value) -> list:
    """Ensure a result is a list of strings."""

    if not isinstance(value, list):
        return []

    return [
        str(item).strip()
        for item in value
        if str(item).strip()
    ]


def normalize_result(data: dict) -> dict:
    """Make sure the AI response contains the fields the app needs."""

    if not isinstance(data, dict):
        raise ValueError("The AI returned an unexpected response format.")

    breakdown_raw = data.get("score_breakdown") or {}

    breakdown = {
        key: clamp_score(breakdown_raw.get(key))
        for key in SCORE_KEYS
    }

    default_score = round(
        sum(breakdown.values()) / len(breakdown)
    )

    overall_score = clamp_score(
        data.get("overall_score"),
        default=default_score,
    )

    improvements = []

    for item in data.get("improvements", []):
        if not isinstance(item, dict):
            continue

        priority = str(item.get("priority", "Medium")).title()

        if priority not in ["High", "Medium", "Low"]:
            priority = "Medium"

        improvements.append({
            "priority": priority,
            "section": str(item.get("section", "General")),
            "issue": str(item.get("issue", "")),
            "suggestion": str(item.get("suggestion", "")),
        })

    priority_order = {
        "High": 0,
        "Medium": 1,
        "Low": 2,
    }

    improvements.sort(
        key=lambda item: priority_order[item["priority"]]
    )

    rewrites = []

    for item in data.get("rewrite_examples", []):
        if isinstance(item, dict):
            original = item.get("original")
            improved = item.get("improved")

            if original and improved:
                rewrites.append({
                    "original": str(original),
                    "improved": str(improved),
                })

    return {
        "overall_score": overall_score,
        "score_breakdown": breakdown,
        "summary": str(data.get("summary", "")),
        "strengths": string_list(data.get("strengths")),
        "weaknesses": string_list(data.get("weaknesses")),
        "matched_keywords": string_list(data.get("matched_keywords")),
        "missing_keywords": string_list(data.get("missing_keywords")),
        "improvements": improvements,
        "rewrite_examples": rewrites,
    }


# -------------------- GEMINI ANALYSIS --------------------

def analyze_resume(
    resume_text: str,
    job_description: str,
    api_key: str,
    model: str,
) -> dict:

    client = genai.Client(api_key=api_key)

    prompt = build_prompt(
        resume_text[:MAX_RESUME_CHARS],
        job_description[:8000],
    )

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json",
        ),
    )

    return normalize_result(
        parse_json_response(response.text)
    )


def friendly_error(exc: Exception) -> str:
    """Show a simpler message for common API errors."""

    message = str(exc)
    lower_message = message.lower()

    if any(term in lower_message for term in [
        "api key", "api_key", "permission", "401", "403"
    ]):
        return (
            "Your Gemini API key may be invalid or lack permission. "
            "Check the GEMINI_API_KEY value in Streamlit Secrets."
        )

    if any(term in lower_message for term in [
        "429", "quota", "resource_exhausted", "rate limit"
    ]):
        return (
            "The Gemini API limit may have been reached. "
            "Wait a little and try again, or check your API quota."
        )

    if any(term in lower_message for term in [
        "404", "not found", "not supported"
    ]):
        return (
            f"The selected Gemini model may not be available for your "
            f"API key. Select another model in the sidebar. "
            f"Details: {message[:250]}"
        )

    if "503" in lower_message or "unavailable" in lower_message:
        return (
            "Gemini is temporarily unavailable or busy. "
            "Please wait a little and try again."
        )

    return f"An error occurred: {message[:500]}"


# -------------------- UI HELPERS --------------------

def score_label(score: int):

    if score >= 80:
        return "Excellent", "🟢"

    if score >= 65:
        return "Good", "🟡"

    if score >= 50:
        return "Needs Improvement", "🟠"

    return "Poor", "🔴"


def get_api_key() -> str:
    """Read the API key from Streamlit Secrets or sidebar input."""

    try:
        secret = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        secret = ""

    if secret:
        st.sidebar.success("Gemini API key loaded from Secrets.")
        return str(secret).strip()

    return st.sidebar.text_input(
        "Gemini API key",
        type="password",
        help="Get an API key from https://aistudio.google.com/app/apikey",
    ).strip()


# -------------------- DISPLAY RESULTS --------------------

def render_results(result: dict):

    score = result["overall_score"]
    label, icon = score_label(score)

    st.divider()
    st.header("Your Resume Analysis")

    left, right = st.columns([1, 2])

    with left:
        st.metric("Estimated ATS Score", f"{score}/100")
        st.markdown(f"### {icon} {label}")
        st.progress(score / 100)

    with right:
        st.subheader("Summary")
        st.write(result["summary"] or "No summary was provided.")

    st.caption(
        "This score is an AI-generated estimate. Different employers "
        "and ATS platforms may evaluate resumes differently."
    )

    st.subheader("Score Breakdown")

    columns = st.columns(len(SCORE_KEYS))

    for column, (key, title) in zip(columns, SCORE_KEYS.items()):
        with column:
            value = result["score_breakdown"][key]
            st.metric(title, f"{value}/100")
            st.progress(value / 100)

    strength_column, weakness_column = st.columns(2)

    with strength_column:
        st.subheader("✅ Strengths")

        if result["strengths"]:
            for item in result["strengths"]:
                st.markdown(f"- {item}")
        else:
            st.write("No strengths returned.")

    with weakness_column:
        st.subheader("⚠️ Weaknesses")

        if result["weaknesses"]:
            for item in result["weaknesses"]:
                st.markdown(f"- {item}")
        else:
            st.write("No weaknesses returned.")

    keyword_column1, keyword_column2 = st.columns(2)

    with keyword_column1:
        st.subheader("Matched Keywords")

        if result["matched_keywords"]:
            st.write(", ".join(result["matched_keywords"]))
        else:
            st.write("No matched keywords returned.")

    with keyword_column2:
        st.subheader("Missing Keywords")

        if result["missing_keywords"]:
            st.write(", ".join(result["missing_keywords"]))
        else:
            st.write("No missing keywords returned.")

    st.subheader("🛠️ Recommended Improvements")

    badges = {
        "High": "🔴",
        "Medium": "🟠",
        "Low": "🟢",
    }

    if result["improvements"]:
        for number, item in enumerate(
            result["improvements"], start=1
        ):
            title = (
                f"{badges[item['priority']]} "
                f"{item['priority']} | {item['section']}"
            )

            with st.expander(title, expanded=(number <= 3)):
                st.markdown(f"**Problem:** {item['issue']}")
                st.markdown(f"**Suggested Fix:** {item['suggestion']}")
    else:
        st.info("No improvement suggestions were returned.")

    if result["rewrite_examples"]:
        st.subheader("✍️ Resume Sentence Improvements")

        for example in result["rewrite_examples"]:
            st.markdown(f"**Before:** {example['original']}")
            st.markdown(f"**Improved:** {example['improved']}")
            st.divider()

    report_json = json.dumps(result, indent=2, ensure_ascii=False)

    st.download_button(
        label="⬇️ Download Analysis Report (JSON)",
        data=report_json,
        file_name="ats_resume_report.json",
        mime="application/json",
        use_container_width=True,
    )


# -------------------- MAIN APPLICATION --------------------

def main():

    st.set_page_config(
        page_title="ATS Resume Checker",
        page_icon="📄",
        layout="wide",
    )

    st.title("📄 ATS Resume Checker")

    st.write(
        "Upload your resume to receive an estimated ATS score, "
        "keyword analysis, and practical suggestions."
    )

    with st.sidebar:
        st.header("Settings")
        api_key = get_api_key()

        model = st.selectbox(
            "Gemini Model",
            MODEL_OPTIONS,
            index=0,
        )

        st.caption(
            "Your resume text is sent to Google's Gemini API "
            "for analysis. Do not upload sensitive information "
            "you are not comfortable sharing."
        )

    upload_column, job_column = st.columns(2)

    with upload_column:
        uploaded = st.file_uploader(
            "Upload your resume",
            type=["pdf", "docx", "txt"],
            help="Maximum file size: 5 MB.",
        )

    with job_column:
        job_description = st.text_area(
            "Job Description (Optional)",
            height=150,
            placeholder=(
                "Paste the job description here for a more "
                "targeted resume analysis."
            ),
        )

    if st.button(
        "Analyze Resume",
        type="primary",
        use_container_width=True,
    ):

        if not api_key:
            st.error(
                "Please add your Gemini API key in the sidebar "
                "or in Streamlit Secrets."
            )
            st.stop()

        if uploaded is None:
            st.error("Please upload your resume first.")
            st.stop()

        data = uploaded.getvalue()

        if len(data) > MAX_FILE_MB * 1024 * 1024:
            st.error(
                f"The file is larger than {MAX_FILE_MB} MB. "
                "Please upload a smaller file."
            )
            st.stop()

        try:
            with st.spinner("Reading your resume..."):
                resume_text = extract_text(uploaded.name, data)

        except Exception as exc:
            st.error(f"Could not read the resume: {exc}")
            st.stop()

        if len(resume_text.strip()) < 100:
            st.error(
                "Very little text could be extracted from this file. "
                "If it is a scanned PDF, try a text-based PDF or DOCX. "
                "For a damaged DOCX, open it in Word and save a new copy."
            )
            st.stop()

        try:
            with st.spinner(
                "Analyzing your resume with Gemini. Please wait..."
            ):
                result = analyze_resume(
                    resume_text,
                    job_description,
                    api_key,
                    model,
                )

            st.session_state["ats_result"] = result

        except Exception as exc:
            st.error(friendly_error(exc))
            st.stop()

    if "ats_result" in st.session_state:
        render_results(st.session_state["ats_result"])


if __name__ == "__main__":
    main()
