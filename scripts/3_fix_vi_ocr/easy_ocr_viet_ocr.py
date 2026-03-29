import sys
from pathlib import Path

# Go up to project root
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import os
import json
from pathlib import Path
from tqdm import tqdm
from PIL import Image

import easyocr
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor

from dir_config import *

# -------------------------------
# INIT MODELS (LOAD ONCE)
# -------------------------------
print("🔤 Loading OCR models...")

reader = easyocr.Reader(["vi"])

config = Cfg.load_config_from_name("vgg_transformer")
config["weights"] = os.path.join(
    MODELS_DIR, "viet_ocr/pretrained_weight/vgg_transformer.pth"
)
config["cnn"]["pretrained"] = False

detector = Predictor(config)

print("✅ OCR models loaded")


# -------------------------------
# OCR FUNCTION
# -------------------------------
def get_transcript_from_image(img_path):
    results = reader.readtext(img_path, detail=1, paragraph=False)

    if not results:
        return ""

    img = Image.open(img_path).convert("RGB")

    line_texts = []
    for coords, _, _ in results:
        x_min = min(pt[0] for pt in coords)
        y_min = min(pt[1] for pt in coords)
        x_max = max(pt[0] for pt in coords)
        y_max = max(pt[1] for pt in coords)

        cropped = img.crop((x_min, y_min, x_max, y_max))

        try:
            text_viet = detector.predict(cropped)
        except Exception as e:
            print(f"⚠️ VietOCR failed: {e}")
            text_viet = ""

        if text_viet.strip():
            line_texts.append(text_viet)

    return " ".join(line_texts)


# -------------------------------
# CHECK RESUME
# -------------------------------
def is_chapter_processed(json_output_dir, result_dir):
    if not result_dir.exists():
        return False

    input_files = list(json_output_dir.glob("*.json"))
    output_files = list(result_dir.glob("*.json"))

    return len(input_files) > 0 and len(input_files) == len(output_files)


# -------------------------------
# PROCESS ONE JSON FILE
# -------------------------------
def process_json_file(json_path, cut_bubbles_dir, output_dir):
    base_name = json_path.stem
    cut_page_dir = cut_bubbles_dir / base_name

    if not cut_page_dir.exists():
        print(f"⚠️ No cut bubbles for {base_name}, skipping...")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    new_ocr = []

    for idx, _bbox in enumerate(data["texts"]):
        cut_img_path = cut_page_dir / f"{base_name}_{idx:03}.png"

        if not cut_img_path.exists():
            new_ocr.append(data["ocr"][idx])
            continue

        try:
            vi_text = get_transcript_from_image(str(cut_img_path))
            if not vi_text:
                vi_text = data["ocr"][idx]
        except Exception as e:
            print(f"❌ OCR failed: {cut_img_path} | {e}")
            vi_text = data["ocr"][idx]

        new_ocr.append(vi_text)

    data["ocr"] = new_ocr

    out_path = output_dir / json_path.name
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


# -------------------------------
# PROCESS ONE CHAPTER
# -------------------------------
def process_chapter(chapter_path):
    json_output_dir = chapter_path / JSON_RESULTS
    cut_bubbles_dir = chapter_path / CUT_BUBBLES
    output_dir = chapter_path / FIX_VI_OCR_OUTPUT_EASYOCR_VIETOCR

    if not json_output_dir.exists():
        return

    output_dir.mkdir(exist_ok=True)

    if is_chapter_processed(json_output_dir, output_dir):
        return "⏭️ Skipped"

    json_files = list(json_output_dir.glob("*.json"))

    for json_file in tqdm(
        json_files,
        desc=f"📄 {chapter_path.name}",
        leave=False
    ):
        process_json_file(json_file, cut_bubbles_dir, output_dir)

    return "✅ Done"


# -------------------------------
# GET ALL CHAPTERS FROM RESULT_DIR
# -------------------------------
def get_all_vi_chapters(result_dir):
    chapters = []

    vi_dir = Path(result_dir) / "vi"

    if not vi_dir.exists():
        print("⚠️ No 'vi' folder found in RESULT_DIR")
        return chapters

    for manga in vi_dir.iterdir():
        if not manga.is_dir():
            continue

        for chapter in manga.iterdir():
            if chapter.is_dir():
                chapters.append(chapter)

    return chapters


# -------------------------------
# MAIN
# -------------------------------
def main():
    print("🚀 Starting Vietnamese OCR refinement...")

    chapters = get_all_vi_chapters(RESULT_DIR)
    print(f"Found {len(chapters)} chapters")

    for chapter in tqdm(chapters, desc="Processing chapters"):
        try:
            status = process_chapter(chapter)
            if status:
                tqdm.write(f"{status}: {chapter}")
        except Exception as e:
            tqdm.write(f"❌ Error: {chapter} | {e}")

    print("\n🎉 ALL DONE!")


if __name__ == "__main__":
    main()

# D:/miniconda3/envs/easyocr-vietocr/python.exe d:/Downloads/Tu_Lieu/Cao_Hoc/Master_Thesis/master-thesis/scripts/3_fix_vi_ocr/easy_ocr_viet_ocr.py
