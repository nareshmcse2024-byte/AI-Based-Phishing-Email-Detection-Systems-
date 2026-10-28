from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
ARTIFACT_PATH = MODEL_DIR / "model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
DATABASE_PATH = DATA_DIR / "analysis_history.sqlite3"
SYNTHETIC_DATA_PATH = DATA_DIR / "sample_synthetic_emails.csv"
MAX_UPLOAD_BYTES = 2 * 1024 * 1024
MAX_CSV_ROWS = 5_000
MAX_EMAIL_CHARS = 20_000
MAX_URLS = 50
MAX_URL_CHARS = 2_048
LABELS = ("Likely Phishing", "Suspicious", "Benign")
APP_TITLE = "SignalShield | Email Risk Triage"
