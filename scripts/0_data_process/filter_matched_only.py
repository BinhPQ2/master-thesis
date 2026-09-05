import os
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from dir_config import *

try:
    import imagehash
except Exception:
    imagehash = None

try:
    from skimage.metrics import structural_similarity as ssim
except Exception:
    ssim = None

LOG_FILE = Path(TEST_DATA_MATCHED_ONLY_DIR) / "log.txt"

# -------------------------------
# MATCHED CRITERIA CONFIG (TUNE HERE)
# -------------------------------
SSIM_THRESHOLD = 0.75
HIST_CORRELATION_THRESHOLD = 0.90

HIST_CORRELATION_FALLBACK = 0.98
ORB_MATCH_RATIO_FALLBACK = 0.15


# -------------------------------
# COMPARED METRICS
# -------------------------------
def _read_image_cv(path, flags=cv2.IMREAD_COLOR):
    img = cv2.imread(path, flags)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def _to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _resize_to_min(img1, img2):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h, w = min(h1, h2), min(w1, w2)
    return cv2.resize(img1, (w, h)), cv2.resize(img2, (w, h))


def mse(imgA, imgB):
    imgA, imgB = _resize_to_min(imgA, imgB)
    if imgA.ndim == 3:
        imgA = _to_gray(imgA)
        imgB = _to_gray(imgB)
    err = np.mean((imgA.astype("float") - imgB.astype("float")) ** 2)
    return float(err)


def structural_similarity(imgA, imgB):
    if ssim is None:
        raise RuntimeError(
            "skimage is required for SSIM. Install scikit-image."
        )
    imgA, imgB = _resize_to_min(imgA, imgB)
    imgA = _to_gray(imgA)
    imgB = _to_gray(imgB)
    score, _ = ssim(imgA, imgB, full=True)
    return float(score)


def phash_similarity(pathA, pathB):
    if imagehash is None:
        raise RuntimeError(
            "imagehash (and PIL) required for perceptual hash. Install imagehash."
        )
    ha = imagehash.phash(Image.open(pathA))
    hb = imagehash.phash(Image.open(pathB))
    max_bits = ha.hash.size
    hamming = ha - hb
    sim = 1.0 - (hamming / max_bits)
    return float(sim)


def hist_correlation(imgA, imgB, bins=32):
    imgA = cv2.cvtColor(imgA, cv2.COLOR_BGR2HSV)
    imgB = cv2.cvtColor(imgB, cv2.COLOR_BGR2HSV)
    imgA, imgB = _resize_to_min(imgA, imgB)
    histA = cv2.calcHist([imgA], [0, 1], None, [bins, bins], [0, 180, 0, 256])
    histB = cv2.calcHist([imgB], [0, 1], None, [bins, bins], [0, 180, 0, 256])
    cv2.normalize(histA, histA)
    cv2.normalize(histB, histB)
    score = cv2.compareHist(histA, histB, cv2.HISTCMP_CORREL)
    return float(score)


def orb_match_ratio(imgA, imgB, max_features=500):
    imgA_gray = _to_gray(imgA)
    imgB_gray = _to_gray(imgB)
    orb = cv2.ORB_create(nfeatures=max_features)
    kp1, des1 = orb.detectAndCompute(imgA_gray, None)
    kp2, des2 = orb.detectAndCompute(imgB_gray, None)
    if des1 is None or des2 is None:
        return 0.0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    if not matches:
        return 0.0
    ratio = len(matches) / max(1, min(len(kp1), len(kp2)))
    return float(ratio)


def compare_images(pathA, pathB):
    if not os.path.exists(pathA) or not os.path.exists(pathB):
        raise FileNotFoundError("One or both image paths do not exist.")
    imgA = _read_image_cv(pathA, cv2.IMREAD_COLOR)
    imgB = _read_image_cv(pathB, cv2.IMREAD_COLOR)

    results = {}
    results["mse"] = mse(imgA, imgB)

    if ssim is not None:
        results["ssim"] = structural_similarity(imgA, imgB)
    else:
        results["ssim"] = None
    try:
        results["phash"] = phash_similarity(pathA, pathB)
    except Exception:
        results["phash"] = None
    results["hist_correlation"] = hist_correlation(imgA, imgB)
    results["orb_match_ratio"] = orb_match_ratio(imgA, imgB)

    return results


# -------------------------------
# GET ALL MANGA
# -------------------------------
def get_all_manga(data_dir):
    manga_list = []

    en_dir = Path(data_dir) / "en"

    for manga in sorted(en_dir.iterdir()):
        if not manga.is_dir():
            continue

        for chapter in sorted(manga.iterdir()):
            if chapter.is_dir():
                manga_list.append((manga.name, chapter.name))

    return manga_list


# -------------------------------
# FILTER MANGA BY RANGE
# -------------------------------
def filter_by_range(item_list, input_range):
    """Slices a list based on input_range bounds.

    - [] -> returns whole list
    - [start, end] -> returns item_list[start:end]
    - [start] -> returns item_list[start:]
    """
    if not input_range:
        return item_list

    if len(input_range) == 1:
        start = input_range[0]
        return item_list[start:]

    start, end = input_range[0], input_range[1]
    return item_list[start:end]


# -------------------------------
# GET SORTED IMAGES
# -------------------------------
def get_sorted_images(folder):
    files = [
        f
        for f in os.listdir(folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    files.sort()
    return files


# -------------------------------
# CHECK MATCH (TWO-PASS LOGIC)
# -------------------------------
def is_match(en_path, vi_path):
    try:
        result = compare_images(en_path, vi_path)

        ssim_score = result.get("ssim", 0)
        hist_correlation_score = result.get("hist_correlation", 0)
        orb_ratio_score = result.get("orb_match_ratio", 0)

        # Pass 1: Strict match
        if ssim_score is not None and hist_correlation_score is not None:
            if (
                ssim_score >= SSIM_THRESHOLD
                and hist_correlation_score >= HIST_CORRELATION_THRESHOLD
            ):
                return True

        # Pass 2: Fallback condition for translated pages
        if hist_correlation_score is not None and orb_ratio_score is not None:
            if (
                hist_correlation_score >= HIST_CORRELATION_FALLBACK
                and orb_ratio_score >= ORB_MATCH_RATIO_FALLBACK
            ):
                return True

        return False

    except Exception as e:
        print(f"⚠️ Compare error: {e}")
        return False


# -------------------------------
# PROCESS ONE CHAPTER
# -------------------------------
def process_chapter(input_dir, output_dir, manga, chapter, log_file):
    en_path = Path(input_dir) / "en" / manga / chapter
    vi_path = Path(input_dir) / "vi" / manga / chapter

    if not en_path.exists() or not vi_path.exists():
        return

    out_en = Path(output_dir) / "en" / manga / chapter
    out_vi = Path(output_dir) / "vi" / manga / chapter

    out_en.mkdir(parents=True, exist_ok=True)
    out_vi.mkdir(parents=True, exist_ok=True)

    en_images = get_sorted_images(en_path)
    vi_images = get_sorted_images(vi_path)

    vi_idx = 0
    output_idx = 0

    unmatched_en = []
    matched_vi_indices = set()

    for en_img in tqdm(en_images, desc=f"{manga}-{chapter}", leave=False):
        en_full = en_path / en_img
        found = False

        for j in range(vi_idx, len(vi_images)):
            vi_full = vi_path / vi_images[j]

            if is_match(str(en_full), str(vi_full)):
                matched_vi_indices.add(j)
                output_name = f"{output_idx:04d}"
                shutil.copy(en_full, out_en / f"{output_name}{en_full.suffix}")
                shutil.copy(vi_full, out_vi / f"{output_name}{vi_full.suffix}")
                output_idx += 1

                vi_idx = j + 1
                found = True
                break

        if not found:
            unmatched_en.append(en_img)
            print(f"❌ No match for {en_img}")

    unmatched_vi = [
        vi_images[i]
        for i in range(len(vi_images))
        if i not in matched_vi_indices
    ]

    if unmatched_en or unmatched_vi:
        log_file.write(f"{manga} / {chapter}\n")
        log_file.write(f"EN unmatched: {unmatched_en}\n")
        log_file.write(f"VI unmatched: {unmatched_vi}\n\n")


# -------------------------------
# MAIN
# -------------------------------
def main():
    print("🔍 Starting EN-VI alignment...")
    input_dir = RAW_DIR
    output_dir = RAW_UNMATCHED_ONLY_DIR

    # Define range slice: e.g. [0, 20] scans items 0 to 19.
    # Set to [] to scan everything.
    input_range = []

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("UNMATCHED PAGES LOG\n\n")

    manga_list = get_all_manga(TEST_DATA_DIR)
    manga_list = filter_by_range(manga_list, input_range)

    print(f"📋 Processing {len(manga_list)} total items based on range criteria.")

    progress_bar = tqdm(manga_list, desc="Processing chapters")
    for manga, chapter in progress_bar:
        progress_bar.set_description(f"Processing {manga}/{chapter}")
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as log_f:
                process_chapter(input_dir, output_dir, manga, chapter, log_f)
        except Exception as e:
            tqdm.write(f"❌ Error: {manga}/{chapter} | {e}")

    print("\n🎉 DONE!")


if __name__ == "__main__":
    main()