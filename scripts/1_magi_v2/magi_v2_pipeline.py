import sys
from pathlib import Path

# Go up to project root
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import sys

for i, p in enumerate(sys.path):
    print(f"{i}: {p}")

import os
import json
import re
from pathlib import Path
import torch
from transformers import AutoModel
from PIL import Image
import numpy as np
from tqdm import tqdm

from dir_config import *

# -------------------------------
# LOAD MODEL
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on {device}")

model = (
    AutoModel.from_pretrained(
        os.path.join(MODELS_DIR, "magi_model_v2"), trust_remote_code=True
    )
    .cuda()
    .eval()
)


# -------------------------------
# UTIL
# -------------------------------
def read_image(path):
    with open(path, "rb") as f:
        img = Image.open(f).convert("L").convert("RGB")
        return np.array(img)


def create_chapter_pages_and_character_bank(input_folder, character_folder):
    chapter_pages = []
    character_bank = {"images": [], "names": []}

    for file in os.listdir(input_folder):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            match = re.search(r"p(\d+)", file)
            page_number = int(match.group(1)) if match else file
            chapter_pages.append((page_number, file))

    chapter_pages.sort(key=lambda x: x[0])
    chapter_pages = [os.path.join(input_folder, f[1]) for f in chapter_pages]

    for file in os.listdir(character_folder):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            name = file.split("_")[0]
            character_bank["images"].append(os.path.join(character_folder, file))
            character_bank["names"].append(name)

    return chapter_pages, character_bank


# -------------------------------
# GET ALL CHAPTERS
# -------------------------------
def get_all_chapters(data_dir):
    chapters = []

    for language_dir in Path(data_dir).iterdir():
        if not language_dir.is_dir():
            continue

        for manga_dir in language_dir.iterdir():
            if not manga_dir.is_dir():
                continue

            for chapter_dir in manga_dir.iterdir():
                if chapter_dir.is_dir():
                    chapters.append(chapter_dir)

    return chapters


# -------------------------------
# ✅ NEW: CHECK IF ALREADY PROCESSED
# -------------------------------
def is_chapter_processed(data_dir, chapter_path):
    relative_path = chapter_path.relative_to(data_dir)
    output_folder = Path(RESULT_DIR) / relative_path
    json_dir = output_folder / JSON_RESULTS

    if not json_dir.exists():
        return False

    json_files = list(json_dir.glob("*.json"))

    # Count input images
    image_files = [
        f
        for f in os.listdir(chapter_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    # ✅ If counts match → assume done
    return len(json_files) == len(image_files)


# -------------------------------
# PROCESS ONE CHAPTER
# -------------------------------
def process_chapter(data_dir, chapter_path):
    print(f"\n📖 Processing: {chapter_path}")

    relative_path = chapter_path.relative_to(data_dir)
    output_folder = Path(RESULT_DIR) / relative_path
    json_dir = output_folder / JSON_RESULTS
    image_dir = output_folder / MAGI_IMAGE_RESULT

    json_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    chapter_pages_paths, character_bank = create_chapter_pages_and_character_bank(
        chapter_path, CHARACTER_DIR
    )

    if len(chapter_pages_paths) == 0:
        print("⚠️ No images found, skipping...")
        return

    chapter_pages = [read_image(p) for p in chapter_pages_paths]
    character_bank["images"] = [read_image(p) for p in character_bank["images"]]

    with torch.no_grad():
        results = model.do_chapter_wide_prediction(
            chapter_pages, character_bank, use_tqdm=True, do_ocr=True
        )

    for i, (img, res) in enumerate(zip(chapter_pages, results)):
        img_name = Path(chapter_pages_paths[i]).stem

        model.visualise_single_image_prediction(
            img, res, str(image_dir / f"{img_name}.png")
        )

        with open(json_dir / f"{img_name}.json", "w") as f:
            json.dump(res, f, indent=4)

    print(f"✅ Done: {chapter_path}")


# -------------------------------
# MAIN
# -------------------------------
def main():
    print("🚀 Starting batch processing...")
    data_dir = RAW_DIR
    chapters = get_all_chapters(data_dir)
    print(f"Found {len(chapters)} chapters")

    # ✅ Progress bar here
    for chapter in tqdm(chapters, desc="Processing chapters"):
        try:
            # ✅ Skip processed
            if is_chapter_processed(data_dir=data_dir, chapter_path=chapter):
                tqdm.write(f"⏭️ Skipping (already done): {chapter}")
                continue

            process_chapter(data_dir=data_dir, chapter_path=chapter)

        except Exception as e:
            tqdm.write(f"❌ Error in {chapter}: {e}")

    print("\n🎉 ALL DONE!")


if __name__ == "__main__":
    main()
