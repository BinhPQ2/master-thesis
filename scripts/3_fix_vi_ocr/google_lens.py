import sys
import os
import json
import asyncio
from pathlib import Path
from tqdm import tqdm
from chrome_lens_py import LensAPI

# -------------------------------
# IMPORT CONFIG
# -------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from dir_config import *

# -------------------------------
# INIT GOOGLE LENS
# -------------------------------
print("🔤 Initializing Google Lens OCR...")
api = LensAPI()
print("✅ Google Lens ready")


# -------------------------------
# OCR FUNCTION (ASYNC)
# -------------------------------
async def get_transcript_from_image(img_path):
    try:
        result = await api.process_image(image_path=img_path, ocr_language="vi")
        text = result.get("ocr_text", "")
        return " ".join(text.split()) if text.strip() else ""
    except Exception as e:
        print(f"⚠️ Lens OCR failed: {img_path} | {e}")
        return ""


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
# PROCESS ONE JSON FILE (ASYNC)
# -------------------------------
async def process_json_file(json_path, cut_bubbles_dir, output_dir):
    base_name = json_path.stem
    cut_page_dir = cut_bubbles_dir / base_name

    if not cut_page_dir.exists():
        print(f"⚠️ No cut bubbles for {base_name}, skipping...")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    new_ocr = []

    tasks = []
    img_paths = []

    # Prepare async tasks
    for idx, _bbox in enumerate(data["texts"]):
        cut_img_path = cut_page_dir / f"{base_name}_{idx:03}.png"

        if not cut_img_path.exists():
            new_ocr.append(data["ocr"][idx])
            continue

        img_paths.append((idx, cut_img_path))
        tasks.append(get_transcript_from_image(str(cut_img_path)))

    # Run OCR in parallel 🚀
    results = await asyncio.gather(*tasks, return_exceptions=True)

    result_idx = 0
    for idx in range(len(data["texts"])):
        cut_img_path = cut_page_dir / f"{base_name}_{idx:03}.png"

        if not cut_img_path.exists():
            continue

        result = results[result_idx]
        result_idx += 1

        if isinstance(result, Exception) or not result:
            vi_text = data["ocr"][idx]
        else:
            vi_text = result

        new_ocr.append(vi_text)

    data["ocr"] = new_ocr

    out_path = output_dir / json_path.name
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


# -------------------------------
# PROCESS ONE CHAPTER (ASYNC)
# -------------------------------
async def process_chapter(chapter_path):
    json_output_dir = chapter_path / JSON_RESULTS
    cut_bubbles_dir = chapter_path / CUT_BUBBLES
    output_dir = chapter_path / FIX_VI_OCR_OUTPUT_GOOGLE_LENS

    if not json_output_dir.exists():
        return

    output_dir.mkdir(exist_ok=True)

    # Resume check
    if is_chapter_processed(json_output_dir, output_dir):
        return "⏭️ Skipped"

    json_files = list(json_output_dir.glob("*.json"))

    for json_file in tqdm(json_files, desc=f"📄 {chapter_path.name}", leave=False):
        await process_json_file(json_file, cut_bubbles_dir, output_dir)

    return "✅ Done"


# -------------------------------
# GET ALL CHAPTERS
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
    print(f"chapters: {chapters}")
    return chapters


# -------------------------------
# MAIN (ASYNC)
# -------------------------------
async def main():
    print("🚀 Starting Vietnamese OCR refinement (Google Lens)...")

    chapters = get_all_vi_chapters(RESULT_DIR)
    print(f"Found {len(chapters)} chapters")

    for chapter in tqdm(chapters, desc="Processing chapters"):
        try:
            status = await process_chapter(chapter)
            if status:
                tqdm.write(f"{status}: {chapter}")
        except Exception as e:
            tqdm.write(f"❌ Error: {chapter} | {e}")

    print("\n🎉 ALL DONE!")


# -------------------------------
# ENTRY POINT
# -------------------------------
if __name__ == "__main__":
    asyncio.run(main())

# D:/miniconda3/envs/easyocr-vietocr/python.exe d:/Downloads/Tu_Lieu/Cao_Hoc/Master_Thesis/master-thesis/scripts/3_fix_vi_ocr/google_lens.py
