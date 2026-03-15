from pathlib import Path

ROOT = Path(__file__).resolve().parent  # project root
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
TEST_DATA = DATA_DIR / "test_data"
RESULT_DIR = ROOT / "magi_results"
MODELS_DIR = ROOT / "models"
LOG_DIR = ROOT / "logs"
LANGUAGES = ["en", "vi"]

def ensure_dirs(*dirs):
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)