from pathlib import Path

ROOT = Path(__file__).resolve().parent  # project root
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
TEST_DATA_DIR = DATA_DIR / "test_data"
RESULT_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
LOG_DIR = ROOT / "logs"

LANGUAGES = ["en", "vi"]
CHARACTER_DIR = DATA_DIR / "characters"
CUT_BUBBLES = "cut_bubbles"
JSON_RESULTS = "json_results"
MAGI_IMAGE_RESULT = "magi_image_results"


def ensure_dirs(*dirs):
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
