import sys
from pathlib import Path

# -------------------------------
# FIX PATH
# -------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from dir_config import *

import json
import pandas as pd
from tqdm import tqdm
from datetime import datetime

# -------------------------------
# OUTPUT DIR
# -------------------------------
CSV_OUTPUT = Path(RESULT_DIR) / CSV_OUTPUT
CSV_OUTPUT.mkdir(exist_ok=True)


# -------------------------------
# GET ALL VI CHAPTERS
# -------------------------------
def get_all_vi_chapters():
    chapters = []
    vi_dir = Path(RESULT_DIR) / "vi"

    for manga in vi_dir.iterdir():
        if not manga.is_dir():
            continue

        for chapter in manga.iterdir():
            if chapter.is_dir():
                chapters.append(chapter)

    return chapters


# -------------------------------
# PROCESS ONE CHAPTER → CSV
# -------------------------------
def process_chapter(chapter_path):
    relative = chapter_path.relative_to(Path(RESULT_DIR) / "vi")

    manga = relative.parts[0]
    chapter = relative.parts[1]

    en_chapter_path = Path(RESULT_DIR) / "en" / manga / chapter

    en_json_dir = en_chapter_path / JSON_RESULTS
    vi_json_dir = chapter_path / FIX_VI_OCR_OUTPUT_GOOGLE_LENS

    if not en_json_dir.exists() or not vi_json_dir.exists():
        return None

    rows = []

    en_files = list(en_json_dir.glob("*.json"))

    for en_file in en_files:
        base_name = en_file.stem
        vi_file = vi_json_dir / f"{base_name}.json"

        if not vi_file.exists():
            continue

        # load both
        with open(en_file, "r", encoding="utf-8") as f:
            en_data = json.load(f)

        with open(vi_file, "r", encoding="utf-8") as f:
            vi_data = json.load(f)

        en_ocr = en_data.get("ocr", [])
        vi_ocr = vi_data.get("ocr", [])

        # align by index
        max_len = max(len(en_ocr), len(vi_ocr))

        for idx in range(max_len):
            rows.append(
                {
                    "manga": manga,
                    "chapter": chapter,
                    "page": base_name,
                    "bubble_idx": idx,
                    # OCR
                    "en_ocr": en_ocr[idx] if idx < len(en_ocr) else "",
                    "vi_ocr": vi_ocr[idx] if idx < len(vi_ocr) else "",
                    # TEXT BOXES
                    "en_texts": (
                        en_data["texts"][idx] if idx < len(en_data["texts"]) else ""
                    ),
                    "vi_texts": (
                        vi_data["texts"][idx] if idx < len(vi_data["texts"]) else ""
                    ),
                    # PANELS
                    "en_panels": en_data["panels"][0] if en_data["panels"] else "",
                    "vi_panels": vi_data["panels"][0] if vi_data["panels"] else "",
                    # CHARACTERS
                    "en_characters": (
                        en_data["characters"][idx]
                        if idx < len(en_data["characters"])
                        else ""
                    ),
                    "vi_characters": (
                        vi_data["characters"][idx]
                        if idx < len(vi_data["characters"])
                        else ""
                    ),
                    # ESSENTIAL FLAG
                    "en_is_essential": (
                        en_data["is_essential_text"][idx]
                        if idx < len(en_data["is_essential_text"])
                        else ""
                    ),
                    "vi_is_essential": (
                        vi_data["is_essential_text"][idx]
                        if idx < len(vi_data["is_essential_text"])
                        else ""
                    ),
                    # CHARACTER NAME
                    "en_character_name": (
                        en_data["character_names"][idx]
                        if idx < len(en_data["character_names"])
                        else ""
                    ),
                    "vi_character_name": (
                        vi_data["character_names"][idx]
                        if idx < len(vi_data["character_names"])
                        else ""
                    ),
                }
            )

    return rows


# -------------------------------
# MAIN
# -------------------------------
def main():
    print("📊 Building CSV dataset...")

    chapters = get_all_vi_chapters()

    all_rows = []

    for chapter in tqdm(chapters, desc="Processing chapters"):
        try:
            rows = process_chapter(chapter)
            if rows:
                all_rows.extend(rows)
        except Exception as e:
            tqdm.write(f"❌ Error: {chapter} | {e}")

    # -------------------------------
    # SAVE CSV
    # -------------------------------
    df = pd.DataFrame(all_rows)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_csv_name = f"en_vi_dialogues_{timestamp}.csv"
    output_path = CSV_OUTPUT / saved_csv_name
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\n✅ CSV saved to: {output_path}")
    print(f"Total rows: {len(df)}")


if __name__ == "__main__":
    main()
