import os
from pathlib import Path
from dataclasses import dataclass

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent

# Load .env for local development if present
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    db_path: Path
    chroma_dir: Path
    uploads_dir: Path
    gemini_api_key: str
    gemini_chat_model: str
    gemini_embed_model: str


def get_settings() -> Settings:
    """Return application settings derived from environment variables."""
    data_dir = PROJECT_ROOT / "data"
    db_path = Path(os.environ.get("APP_DB_PATH", data_dir / "app.db"))
    chroma_dir = Path(os.environ.get("CHROMA_DIR", data_dir / "chroma"))
    uploads_dir = data_dir / "uploads"

    # Ensure directories exist
    data_dir.mkdir(parents=True, exist_ok=True)
    chroma_dir.mkdir(parents=True, exist_ok=True)
    uploads_dir.mkdir(parents=True, exist_ok=True)

    gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
    if not gemini_api_key:
        # We don't raise here to keep the app importable; callers should validate.
        gemini_api_key = ""

    return Settings(
        db_path=db_path,
        chroma_dir=chroma_dir,
        uploads_dir=uploads_dir,
        gemini_api_key=gemini_api_key,
        gemini_chat_model=os.environ.get("GEMINI_CHAT_MODEL", "gemini-1.5-pro"),
        gemini_embed_model=os.environ.get("GEMINI_EMBED_MODEL", "text-embedding-004"),
    )


settings = get_settings()

