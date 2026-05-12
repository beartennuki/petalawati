from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / os.getenv("DATA_DIR", "data")
UPLOADS_DIR = DATA_DIR / "uploads"
ARTIFACTS_DIR = DATA_DIR / "artifacts"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

PREFECT_API_URL = os.getenv("PREFECT_API_URL", "http://127.0.0.1:4200/api")

ARCHITECTURES = {
    "kernarc":  "KernArc",
    "swiftpan": "SwiftPan",
    "sepfuse":  "SepFuse",
    "dualfuse": "DualFuse",
    "syncgen":  "SyncGen",
}

ARCHITECTURE_KEYS = tuple(ARCHITECTURES.keys())
