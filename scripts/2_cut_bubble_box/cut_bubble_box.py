import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

# -------------------------------
# IMPORTS
# -------------------------------
import os
import json
from PIL import Image
from tqdm import tqdm

from dir_config import *


# -------------------------------
# GET ALL VI CHAPTERS
# -------------------------------
def get_all_chapters(result_dir):
    chapters = []

    vi_dir = Path(result_dir) / "vi"

    if not vi_dir.exists():
        return chapters

    for manga in vi_dir.iterdir():
        if not manga.is_dir():
            continue

        for chapter in manga.iterdir():
            if not chapter.is_dir():
                continue

            if (chapter / JSON_RESULTS).exists():
                chapters.append(chapter)

    return chapters


# -------------------------------
# RESUME CHECK
# -------------------------------
def is_chapter_processed(json_dir, cut_dir):
    if not cut_dir.exists():
        return False

    json_files = list(json_dir.glob("*.json"))
    existing_pages = [d for d in cut_dir.iterdir() if d.is_dir()]

    # simple check: number of page folders == number of json files
    return len(existing_pages) == len(json_files)


# -------------------------------
# PROCESS ONE CHAPTER
# -------------------------------
def process_chapter(chapter_path):
    relative_path = chapter_path.relative_to(RESULT_DIR)

    json_dir = chapter_path / JSON_RESULTS
    cut_dir = chapter_path / CUT_BUBBLES

    # find corresponding raw images
    raw_chapter_path = Path(TEST_DATA_DIR) / relative_path

    if not raw_chapter_path.exists():
        print(f"⚠️ Raw images not found: {raw_chapter_path}")
        return "❌ Missing raw"

    cut_dir.mkdir(exist_ok=True)

    # resume check
    if is_chapter_processed(json_dir, cut_dir):
        return "⏭️ Skipped"

    raw_images = os.listdir(raw_chapter_path)

    for json_file in json_dir.glob("*.json"):
        base_name = json_file.stem

        # find matching image
        image_candidates = [
            f
            for f in raw_images
            if f.startswith(base_name) and f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        if not image_candidates:
            print(f"⚠️ No image for {base_name}")
            continue

        image_path = raw_chapter_path / image_candidates[0]

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        image = Image.open(image_path).convert("RGB")

        page_dir = cut_dir / base_name
        page_dir.mkdir(exist_ok=True)

        for idx, bbox in enumerate(data["texts"]):
            x1, y1, x2, y2 = map(int, bbox)

            # expand box
            w, h = x2 - x1, y2 - y1
            dw, dh = int(w * 0.1), int(h * 0.1)

            x1 = max(0, x1 - dw)
            y1 = max(0, y1 - dh)
            x2 = min(image.width, x2 + dw)
            y2 = min(image.height, y2 + dh)

            cropped = image.crop((x1, y1, x2, y2))

            out_path = page_dir / f"{base_name}_{idx:03}.png"
            cropped.convert("L").save(out_path)

    return "✅ Done"


# -------------------------------
# MAIN
# -------------------------------
def main():
    print("✂️ Starting bubble cutting...")

    chapters = get_all_chapters(RESULT_DIR)
    print(f"Found {len(chapters)} chapters")

    for chapter in tqdm(chapters, desc="Cutting bubbles"):
        try:
            status = process_chapter(chapter)
            if status:
                tqdm.write(f"{status}: {chapter}")
        except Exception as e:
            tqdm.write(f"❌ Error: {chapter} | {e}")

    print("\n🎉 DONE cutting bubbles!")


if __name__ == "__main__":
    main()
