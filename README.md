# 📄 ATS Resume Checker

Upload a resume (PDF, DOCX or TXT) and get an **ATS score out of 100**, a score breakdown,
matched/missing keywords, prioritised improvements and example rewrites.
Built with **Streamlit** and **Google Gemini Flash**.

## Features
- Reads PDF, DOCX (including tables) and TXT resumes
- Optional job description for targeted keyword matching
- Overall ATS score + 5 category scores
- Strengths, weaknesses, matched / missing keywords
- Prioritised improvements (High / Medium / Low) and before/after rewrites
- Downloadable JSON report
- Friendly error messages (bad API key, quota, scanned PDFs, etc.)

## Project structure
```
ats-resume-checker/
├── app.py
├── requirements.txt
└── README.md
```

## Run locally
1. Get a free API key: https://aistudio.google.com/app/apikey
2. Install and run:
   ```bash
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   streamlit run app.py
   ```
3. Paste your API key in the sidebar (or see below to store it as a secret).

### Optional: store the key in a local secrets file
Create `.streamlit/secrets.toml` (never commit this file):
```toml
GEMINI_API_KEY = "your-key-here"
```

## Deploy on Streamlit Community Cloud
1. Push this project to a **GitHub** repository (see steps below).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app** → choose your repo, branch `main`, main file `app.py`.
4. Open **Advanced settings → Secrets** and paste:
   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```
5. Click **Deploy**.

## Notes
- Model names can change. If you get a "model not found" error, pick another one in the sidebar
  or update `MODEL_OPTIONS` in `app.py`.
- The ATS score is an AI estimate, not the output of a real ATS system.
- Resume text is sent to Google's Gemini API for analysis.
