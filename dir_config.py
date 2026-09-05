from pathlib import Path

from torch import fix_

ROOT = Path(__file__).resolve().parent  # project root
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
TEST_DATA_DIR = DATA_DIR / "test_data"
RESULT_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"

LANGUAGES = ["en", "vi"]
CHARACTER_DIR = DATA_DIR / "characters"
CUT_BUBBLES = "cut_bubbles"
JSON_RESULTS = "json_results"
MAGI_IMAGE_RESULT = "magi_image_results"
FIX_VI_OCR_OUTPUT_EASYOCR_VIETOCR = "easy_ocr_viet_ocr_result"
FIX_VI_OCR_OUTPUT_GOOGLE_LENS = "google_lens_ocr_result"
CSV_OUTPUT = "csv_output"
RAW_UNMATCHED_ONLY_DIR = DATA_DIR / "raw_unmatched_only"
TEST_DATA_MATCHED_ONLY_DIR = DATA_DIR / "test_data_matched_only"


def ensure_dirs(*dirs):
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
