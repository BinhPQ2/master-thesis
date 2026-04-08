from ast import Import
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from dir_config import *

import os
import shutil
from tqdm import tqdm
import cv2
import numpy as np

try:
    import imagehash
except Exception:
    imagehash = None

try:
    from skimage.metrics import structural_similarity as ssim
except Exception:
    ssim = None
LOG_FILE = Path(TEST_DATA_UNDUPED_DIR) / "log.txt"

# -------------------------------
# CONFIG (TUNE HERE)
# -------------------------------
SSIM_THRESHOLD = 0.75
HIST_CORRELATION_THRESHOLD = 0.9


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
        raise RuntimeError("skimage is required for SSIM. Install scikit-image.")
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
    # normalized similarity: 1 - (hamming / hash_size)
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
    # ratio of good matches to min(keypoints)
    ratio = len(matches) / max(1, min(len(kp1), len(kp2)))
    return float(ratio)


def compare_images(pathA, pathB):
    if not os.path.exists(pathA) or not os.path.exists(pathB):
        raise FileNotFoundError("One or both image paths do not exist.")
    imgA = _read_image_cv(pathA, cv2.IMREAD_COLOR)
    imgB = _read_image_cv(pathB, cv2.IMREAD_COLOR)

    results = {}
    results["mse"] = mse(imgA, imgB)  # lower is more similar

    if ssim is not None:
        results["ssim"] = structural_similarity(imgA, imgB)  # 1.0 is identical
    else:
        results["ssim"] = None
    try:
        results["phash"] = phash_similarity(pathA, pathB)  # 1.0 is identical
    except Exception:
        results["phash"] = None
    results["hist_correlation"] = hist_correlation(imgA, imgB)  # 1.0 is identical
    results["orb_match_ratio"] = orb_match_ratio(imgA, imgB)  # 0..1 ratio

    return results


# -------------------------------
# GET ALL MANGA
# -------------------------------
def get_all_manga(data_dir):
    manga_list = []

    en_dir = Path(data_dir) / "en"

    for manga in en_dir.iterdir():
        if not manga.is_dir():
            continue

        for chapter in manga.iterdir():
            if chapter.is_dir():
                manga_list.append((manga.name, chapter.name))

    return manga_list


# -------------------------------
# GET SORTED IMAGES
# -------------------------------
def get_sorted_images(folder):
    files = [
        f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    # sort by filename (assuming p001, p002...)
    files.sort()
    return files


# -------------------------------
# CHECK MATCH
# -------------------------------
def is_match(en_path, vi_path):
    try:
        result = compare_images(en_path, vi_path)

        ssim_score = result.get("ssim", 0)
        hist_correlation_score = result.get("hist_correlation", 0)

        if ssim_score is None or hist_correlation_score is None:
            return False

        return (
            ssim_score >= SSIM_THRESHOLD
            and hist_correlation_score >= HIST_CORRELATION_THRESHOLD
        )

    except Exception as e:
        print(f"⚠️ Compare error: {e}")
        return False


# -------------------------------
# PROCESS ONE CHAPTER
# -------------------------------
def process_chapter(manga, chapter, log_file):
    en_path = Path(TEST_DATA_DIR) / "en" / manga / chapter
    vi_path = Path(TEST_DATA_DIR) / "vi" / manga / chapter

    if not en_path.exists() or not vi_path.exists():
        return

    out_en = Path(TEST_DATA_UNDUPED_DIR) / "en" / manga / chapter
    out_vi = Path(TEST_DATA_UNDUPED_DIR) / "vi" / manga / chapter

    out_en.mkdir(parents=True, exist_ok=True)
    out_vi.mkdir(parents=True, exist_ok=True)

    en_images = get_sorted_images(en_path)
    vi_images = get_sorted_images(vi_path)

    vi_idx = 0

    unmatched_en = []
    matched_vi_indices = set()

    for en_img in tqdm(en_images, desc=f"{manga}-{chapter}", leave=False):
        en_full = en_path / en_img

        found = False

        for j in range(vi_idx, len(vi_images)):
            vi_full = vi_path / vi_images[j]

            if is_match(str(en_full), str(vi_full)):
                # ✅ MATCH FOUND
                matched_vi_indices.add(j)
                shutil.copy(en_full, out_en / en_img)
                shutil.copy(vi_full, out_vi / vi_images[j])

                vi_idx = j + 1  # move forward
                found = True
                break

        if not found:
            unmatched_en.append(en_img)
            print(f"❌ No match for {en_img}")

    unmatched_vi = [
        vi_images[i] for i in range(len(vi_images)) if i not in matched_vi_indices
    ]

    if unmatched_en or unmatched_vi:
        log_file.write(f"{manga} / {chapter}\n")
        log_file.write(f"EN unmatched: {unmatched_en}\n")
        log_file.write(f"VI unmatched: {unmatched_vi}\n\n")


# -------------------------------
# MAIN
# -------------------------------
def main():
    # reset log file
    print("🔍 Starting EN-VI alignment...")

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("UNMATCHED PAGES LOG\n\n")

    manga_list = get_all_manga(TEST_DATA_DIR)

    progress_bar = tqdm(manga_list, desc="Processing chapters")
    for manga, chapter in progress_bar:
        progress_bar.set_description(f"Processing {manga}/{chapter}")
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as log_f:
                process_chapter(manga, chapter, log_f)
        except Exception as e:
            tqdm.write(f"❌ Error: {manga}/{chapter} | {e}")

    print("\n🎉 DONE!")


if __name__ == "__main__":
    main()
