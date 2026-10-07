"""
ATS Resume Checker
Upload a resume (PDF / DOCX / TXT) -> get an ATS score + concrete improvements.
UI: Streamlit | AI: Google Gemini Flash
"""

import json
import re
from io import BytesIO

import streamlit as st
from docx import Document
from google import genai
from google.genai import types
from pypdf import PdfReader

# ----------------------------- Config ---------------------------------------
MODEL_OPTIONS = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-flash-latest"]
MAX_RESUME_CHARS = 20_000  # keeps the prompt small and fast
MAX_FILE_MB = 5

SCORE_KEYS = {
    "keywords_match": "Keywords & Skills",
    "formatting": "Formatting & ATS Parsability",
    "structure": "Sections & Structure",
    "impact": "Achievements & Impact",
    "readability": "Clarity & Readability",
}


# ----------------------------- File parsing ---------------------------------
def extract_text(file_name: str, data: bytes) -> str:
    """Extract plain text from a PDF, DOCX or TXT upload."""
    name = file_name.lower()

    if name.endswith(".pdf"):
        reader = PdfReader(BytesIO(data))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                raise ValueError("This PDF is password-protected. Please upload an unlocked copy.")
        pages = [(page.extract_text() or "") for page in reader.pages]
        return "\n".join(pages).strip()

    if name.endswith(".docx"):
        doc = Document(BytesIO(data))
        parts = [p.text for p in doc.paragraphs if p.text.strip()]
        # Many resumes keep content inside tables
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    if cell.text.strip():
                        parts.append(cell.text.strip())
        return "\n".join(parts).strip()

    if name.endswith(".txt"):
        return data.decode("utf-8", errors="ignore").strip()

    raise ValueError("Unsupported file type. Please upload a PDF, DOCX or TXT file.")


# ----------------------------- AI layer -------------------------------------
def build_prompt(resume_text: str, job_description: str) -> str:
    jd_block = (
        f"JOB DESCRIPTION:\n{job_description.strip()}"
        if job_description.strip()
        else "JOB DESCRIPTION: (not provided - evaluate against general ATS best practices "
        "and infer the most likely target role from the resume)"
    )
    return f"""You are an expert ATS (Applicant Tracking System) analyst and professional resume coach.
Analyse the resume below and return ONLY a valid JSON object (no markdown, no commentary)
with exactly this schema:

{{
  "overall_score": <integer 0-100>,
  "score_breakdown": {{
    "keywords_match": <integer 0-100>,
    "formatting": <integer 0-100>,
    "structure": <integer 0-100>,
    "impact": <integer 0-100>,
    "readability": <integer 0-100>
  }},
  "summary": "<2-3 sentence honest assessment>",
  "strengths": ["<string>", ...],
  "weaknesses": ["<string>", ...],
  "matched_keywords": ["<string>", ...],
  "missing_keywords": ["<string>", ...],
  "improvements": [
    {{"priority": "High" | "Medium" | "Low", "section": "<resume section>", "issue": "<what is wrong>", "suggestion": "<specific fix>"}}
  ],
  "rewrite_examples": [
    {{"original": "<weak line copied from the resume>", "improved": "<stronger ATS-friendly version>"}}
  ]
}}

Scoring rules:
- Be strict and realistic. 90+ is rare. Most resumes score 50-80.
- If a job description is given, keyword matching must be judged against it.
- Penalise: missing contact info, missing sections (summary, skills, experience, education),
  no quantified achievements, tables/columns/graphics signs, vague bullets, typos, very long paragraphs.
- Give 5-10 improvements sorted by priority (High first) and 2-4 rewrite examples.
- Never invent experience the candidate does not have; rewrites must reuse real facts from the resume.

{jd_block}

RESUME:
{resume_text}
"""


def parse_json_response(raw: str) -> dict:
    """Robustly parse JSON, even if the model wraps it in markdown fences or extra text."""
    if not raw:
        raise ValueError("The model returned an empty response.")
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise ValueError("Could not parse the model response as JSON.")


def _clamp(value, default=0) -> int:
    try:
        return max(0, min(100, int(round(float(value)))))
    except (TypeError, ValueError):
        return default


def _str_list(value) -> list:
    if not isinstance(value, list):
        return []
    return [str(v).strip() for v in value if str(v).strip()]


def normalize_result(data: dict) -> dict:
    """Guarantee every field exists with the right type so the UI never crashes."""
    if not isinstance(data, dict):
        raise ValueError("Unexpected response format from the model.")

    breakdown_raw = data.get("score_breakdown") or {}
    breakdown = {k: _clamp(breakdown_raw.get(k)) for k in SCORE_KEYS}

    overall = data.get("overall_score")
    overall = _clamp(overall, default=round(sum(breakdown.values()) / len(breakdown)))

    improvements = []
    for item in data.get("improvements") or []:
        if isinstance(item, dict):
            priority = str(item.get("priority", "Medium")).title()
            if priority not in ("High", "Medium", "Low"):
                priority = "Medium"
            improvements.append(
                {
                    "priority": priority,
                    "section": str(item.get("section", "General")),
                    "issue": str(item.get("issue", "")),
                    "suggestion": str(item.get("suggestion", "")),
                }
            )
    order = {"High": 0, "Medium": 1, "Low": 2}
    improvements.sort(key=lambda x: order[x["priority"]])

    rewrites = []
    for item in data.get("rewrite_examples") or []:
        if isinstance(item, dict) and item.get("original") and item.get("improved"):
            rewrites.append({"original": str(item["original"]), "improved": str(item["improved"])})

    return {
        "overall_score": overall,
        "score_breakdown": breakdown,
        "summary": str(data.get("summary", "")),
        "strengths": _str_list(data.get("strengths")),
        "weaknesses": _str_list(data.get("weaknesses")),
        "matched_keywords": _str_list(data.get("matched_keywords")),
        "missing_keywords": _str_list(data.get("missing_keywords")),
        "improvements": improvements,
        "rewrite_examples": rewrites,
    }


def analyze_resume(resume_text: str, job_description: str, api_key: str, model: str) -> dict:
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=build_prompt(resume_text[:MAX_RESUME_CHARS], job_description[:8000]),
        config=types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json",
        ),
    )
    return normalize_result(parse_json_response(response.text))


def friendly_error(exc: Exception) -> str:
    msg = str(exc)
    low = msg.lower()
    if "api key" in low or "api_key" in low or "permission" in low or "401" in low or "403" in low:
        return "Your Gemini API key looks invalid or lacks permission. Please check it."
    if "429" in low or "quota" in low or "resource_exhausted" in low:
        return "Rate limit / quota reached. Wait a minute and try again, or use another key."
    if "404" in low or "not found" in low:
        return "That model name is not available for your key. Pick another model in the sidebar."
    return f"Something went wrong: {msg}"


# ----------------------------- UI helpers -----------------------------------
def score_label(score: int):
    if score >= 80:
        return "Excellent", "🟢"
    if score >= 65:
        return "Good", "🟡"
    if score >= 50:
        return "Needs work", "🟠"
    return "Poor", "🔴"


def get_api_key() -> str:
    """Prefer the key from Streamlit secrets (deployed), else the sidebar input."""
    try:
        secret = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:  # no secrets file locally
        secret = ""
    if secret:
        st.sidebar.success("API key loaded from secrets ✅")
        return secret
    return st.sidebar.text_input(
        "Gemini API key",
        type="password",
        help="Free key: https://aistudio.google.com/app/apikey",
    ).strip()


def render_results(r: dict):
    score = r["overall_score"]
    label, icon = score_label(score)

    st.divider()
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("ATS Score", f"{score} / 100")
        st.markdown(f"### {icon} {label}")
        st.progress(score / 100)
    with c2:
        st.subheader("Summary")
        st.write(r["summary"] or "No summary returned.")

    st.subheader("Score breakdown")
    cols = st.columns(len(SCORE_KEYS))
    for col, (key, title) in zip(cols, SCORE_KEYS.items()):
        with col:
            st.metric(title, r["score_breakdown"][key])
            st.progress(r["score_breakdown"][key] / 100)

    s_col, w_col = st.columns(2)
    with s_col:
        st.subheader("✅ Strengths")
        for s in r["strengths"] or ["—"]:
            st.markdown(f"- {s}")
    with w_col:
        st.subheader("⚠️ Weaknesses")
        for w in r["weaknesses"] or ["—"]:
            st.markdown(f"- {w}")

    k1, k2 = st.columns(2)
    with k1:
        st.subheader("Matched keywords")
        st.write(", ".join(f"`{k}`" for k in r["matched_keywords"]) or "—")
    with k2:
        st.subheader("Missing keywords")
        st.write(", ".join(f"`{k}`" for k in r["missing_keywords"]) or "—")

    st.subheader("🛠️ Recommended improvements")
    badge = {"High": "🔴", "Medium": "🟠", "Low": "🟢"}
    for i, imp in enumerate(r["improvements"], 1):
        with st.expander(f"{badge[imp['priority']]} {imp['priority']} · {imp['section']} — {imp['issue'][:70]}", expanded=(i <= 3)):
            st.markdown(f"**Issue:** {imp['issue']}")
            st.markdown(f"**Fix:** {imp['suggestion']}")
    if not r["improvements"]:
        st.info("No improvements returned.")

    if r["rewrite_examples"]:
        st.subheader("✍️ Example rewrites")
        for ex in r["rewrite_examples"]:
            st.markdown(f"**Before:** {ex['original']}")
            st.markdown(f"**After:** {ex['improved']}")
            st.markdown("---")

    st.download_button(
        "⬇️ Download report (JSON)",
        data=json.dumps(r, indent=2),
        file_name="ats_report.json",
        mime="application/json",
    )


# ----------------------------- Main app -------------------------------------
def main():
    st.set_page_config(page_title="ATS Resume Checker", page_icon="📄", layout="wide")
    st.title("📄 ATS Resume Checker")
    st.caption("Upload your resume, get an ATS score and specific improvements — powered by Gemini Flash.")

    with st.sidebar:
        st.header("Settings")
    api_key = get_api_key()
    model = st.sidebar.selectbox("Gemini model", MODEL_OPTIONS, index=0)
    st.sidebar.caption("Your resume is sent to Google's Gemini API for analysis. Don't upload anything you aren't comfortable sharing.")

    left, right = st.columns(2)
    with left:
        uploaded = st.file_uploader("Upload resume", type=["pdf", "docx", "txt"])
    with right:
        jd = st.text_area(
            "Job description (optional, recommended)",
            height=150,
            placeholder="Paste the job description to get a targeted keyword match...",
        )

    if st.button("Analyze resume", type="primary", use_container_width=True):
        if not api_key:
            st.error("Please enter your Gemini API key in the sidebar.")
            st.stop()
        if uploaded is None:
            st.error("Please upload a resume first.")
            st.stop()

        data = uploaded.getvalue()
        if len(data) > MAX_FILE_MB * 1024 * 1024:
            st.error(f"File is larger than {MAX_FILE_MB} MB.")
            st.stop()

        try:
            with st.spinner("Reading your resume..."):
                text = extract_text(uploaded.name, data)
        except Exception as exc:
            st.error(f"Could not read the file: {exc}")
            st.stop()

        if len(text) < 100:
            st.error(
                "Almost no text could be extracted. If this is a scanned/image PDF, "
                "an ATS can't read it either — export a text-based PDF or DOCX instead."
            )
            st.stop()

        try:
            with st.spinner("Analysing with Gemini..."):
                st.session_state["result"] = analyze_resume(text, jd, api_key, model)
        except Exception as exc:
            st.error(friendly_error(exc))
            st.stop()

    if "result" in st.session_state:
        render_results(st.session_state["result"])


if __name__ == "__main__":
    main()
