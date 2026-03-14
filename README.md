# PM Research Assistant (Streamlit + Gemini + Chroma)

This app is a lightweight **product management research assistant**. It lets PMs:

- Upload internal docs and customer feedback.
- Index everything into **SQLite + Chroma** with **Gemini embeddings**.
- Run structured **research briefs** and generate Markdown research reports.

## Quickstart

### 1. Create a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate  # on Windows
# or
source .venv/bin/activate  # on macOS / Linux
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

Create a `.env` file in the project root:

```bash
GEMINI_API_KEY=your_api_key_here
APP_DB_PATH=data/app.db
CHROMA_DIR=data/chroma
GEMINI_CHAT_MODEL=gemini-1.5-pro
GEMINI_EMBED_MODEL=text-embedding-004
```

The app will also create `data/uploads` automatically for raw uploads.

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

Then open the URL printed in the terminal (typically `http://localhost:8501`).

## Project structure

- `app.py` – Streamlit entrypoint, sidebar navigation, and pages.
- `config.py` – Environment-driven configuration (paths, model names).
- `db.py` – SQLite schema and helpers for documents, briefs, and reports.
- `rag_store.py` – Chroma initialization and vector search helpers.
- `llm_client.py` – Gemini chat + embeddings wrapper.
- `scraper.py` – Simple URL + HTML → text and CSV helpers.
- `research_pipeline.py` – Ingestion and research orchestration.
- `data/` – Local data directory:
  - `app.db` – SQLite database (created on first run).
  - `chroma/` – Chroma persistence directory.
  - `uploads/` – Raw uploaded files.

## Status

This is an early version focused on:

- Clear, modular Python code.
- A simple but modern Streamlit UX for PMs.
- Easy local setup with a single process.

Future enhancements (optional) include authentication, better PDF handling, and background jobs for long-running research.

