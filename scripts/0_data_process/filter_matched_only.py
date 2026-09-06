import os
from pathlib import Path
import sys

import cv2
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from dir_config import *

try:
    from skimage.metrics import structural_similarity as ssim
except Exception:
    ssim = None


# -------------------------------
# CONFIG / THRESHOLDS (TUNE HERE)
# -------------------------------
HIST_CORRELATION_THRESHOLD = 0.92  # Minimum HSV histogram similarity for anchors
SSIM_THRESHOLD = 0.75              # Minimum structural similarity for anchors
ANCHOR_SEARCH_DEPTH = 6            # Max pages from start/end to search for anchors
INPUT_RANGE = [10, 100]           #     # Define range slice: e.g. [0, 20] scans items 0 to 19. Set to [] to scan everything.


# -------------------------------
# ANCHOR METRICS & COMPARISON
# -------------------------------
def _to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _resize_to_min(img1, img2):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h, w = min(h1, h2), min(w1, w2)
    return cv2.resize(img1, (w, h)), cv2.resize(img2, (w, h))


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


def structural_similarity(imgA, imgB):
    if ssim is None:
        return 1.0
    imgA, imgB = _resize_to_min(imgA, imgB)
    imgA = _to_gray(imgA)
    imgB = _to_gray(imgB)
    score, _ = ssim(imgA, imgB, full=True)
    return float(score)


def strict_is_match(imgA, imgB):
    """Strict similarity check using top-level threshold configs."""
    try:
        h_score = hist_correlation(imgA, imgB)
        s_score = structural_similarity(imgA, imgB) if ssim else 1.0
        return h_score >= HIST_CORRELATION_THRESHOLD and s_score >= SSIM_THRESHOLD
    except Exception:
        return False


# -------------------------------
# PREPROCESSING & IMAGE SPLITTING
# -------------------------------
def load_and_preprocess_images(folder_path):
    """Loads images, auto-detects wide spreads (width > height),

    and splits them into right and left half-pages in RTL reading order.
    """
    raw_files = [
        f
        for f in os.listdir(folder_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    raw_files.sort()

    processed_pages = []

    for f in raw_files:
        full_path = Path(folder_path) / f
        img = cv2.imread(str(full_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        if w > h:
            half_w = w // 2
            # Manga Reading Order (Right-to-Left):
            # Right half is read first, Left half is read second
            right_half = img[:, half_w:]
            left_half = img[:, :half_w]

            processed_pages.append((f"{Path(f).stem}_r.jpg", right_half))
            processed_pages.append((f"{Path(f).stem}_l.jpg", left_half))
        else:
            processed_pages.append((f, img))

    return processed_pages


# -------------------------------
# BOUNDARY ANCHOR SEARCH
# -------------------------------
def find_start_anchor(en_pages, vi_pages, search_depth=ANCHOR_SEARCH_DEPTH):
    """Scans forward from the beginning to find the first matching story page."""
    max_i = min(search_depth, len(en_pages))
    max_j = min(search_depth, len(vi_pages))

    for i in range(max_i):
        for j in range(max_j):
            if strict_is_match(en_pages[i][1], vi_pages[j][1]):
                return i, j

    return 0, 0


def find_end_anchor(en_pages, vi_pages, search_depth=ANCHOR_SEARCH_DEPTH):
    """Scans backward from the end to find the last matching story page."""
    len_en, len_vi = len(en_pages), len(vi_pages)
    max_i = min(search_depth, len_en)
    max_j = min(search_depth, len_vi)

    for i in range(1, max_i + 1):
        for j in range(1, max_j + 1):
            if strict_is_match(en_pages[-i][1], vi_pages[-j][1]):
                return len_en - i, len_vi - j

    return len_en - 1, len_vi - 1


# -------------------------------
# PROCESS CHAPTER LOGIC
# -------------------------------
def process_chapter(input_dir, output_dir, manga, chapter, manga_index, log_file):
    en_path = Path(input_dir) / "en" / manga / chapter
    vi_path = Path(input_dir) / "vi" / manga / chapter

    if not en_path.exists() or not vi_path.exists():
        return

    out_en = Path(output_dir) / "en" / manga / chapter
    out_vi = Path(output_dir) / "vi" / manga / chapter

    # Pre-process & split double pages in memory
    en_pages = load_and_preprocess_images(en_path)
    vi_pages = load_and_preprocess_images(vi_path)

    if not en_pages or not vi_pages:
        return

    # Find boundary anchors
    en_start, vi_start = find_start_anchor(en_pages, vi_pages)
    en_end, vi_end = find_end_anchor(en_pages, vi_pages)

    # Prevent inverted indices if anchors cross
    if en_start > en_end:
        en_start, en_end = 0, len(en_pages) - 1
    if vi_start > vi_end:
        vi_start, vi_end = 0, len(vi_pages) - 1

    # Slice out outer credits/covers
    trimmed_en = en_pages[en_start : en_end + 1]
    trimmed_vi = vi_pages[vi_start : vi_end + 1]

    out_en.mkdir(parents=True, exist_ok=True)
    out_vi.mkdir(parents=True, exist_ok=True)

    # Case 1: Page counts match perfectly after trimming
    if len(trimmed_en) == len(trimmed_vi):
        for idx, ((_, img_en), (_, img_vi)) in enumerate(zip(trimmed_en, trimmed_vi)):
            output_name = f"{idx:04d}.jpg"
            cv2.imwrite(str(out_en / output_name), img_en)
            cv2.imwrite(str(out_vi / output_name), img_vi)

    # Case 2: Length mismatch remains (Fail-Safe Copy + Log)
    else:
        log_file.write(f"Manga index: {manga_index} | {manga} / {chapter}\n")
        log_file.write(
            f"⚠️ LENGTH MISMATCH: EN={len(trimmed_en)} vs VI={len(trimmed_vi)}\n"
        )
        log_file.write(
            f"EN Anchors: [{en_start}:{en_end}] | VI Anchors: [{vi_start}:{vi_end}]\n\n"
        )

        max_pair_idx = min(len(trimmed_en), len(trimmed_vi))

        # Copy aligned overlapping pairs
        for idx in range(max_pair_idx):
            output_name = f"{idx:04d}.jpg"
            cv2.imwrite(str(out_en / output_name), trimmed_en[idx][1])
            cv2.imwrite(str(out_vi / output_name), trimmed_vi[idx][1])

        # Copy excess unmapped tail pages for manual inspection
        if len(trimmed_en) > max_pair_idx:
            for idx in range(max_pair_idx, len(trimmed_en)):
                cv2.imwrite(str(out_en / f"{idx:04d}_unmatched.jpg"), trimmed_en[idx][1])

        if len(trimmed_vi) > max_pair_idx:
            for idx in range(max_pair_idx, len(trimmed_vi)):
                cv2.imwrite(str(out_vi / f"{idx:04d}_unmatched.jpg"), trimmed_vi[idx][1])


# -------------------------------
# MANGA RETRIEVAL & FILTERING
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


def filter_by_range(item_list, input_range):
    if not input_range:
        return item_list
    if len(input_range) == 1:
        return item_list[input_range[0] :]
    return item_list[input_range[0] : input_range[1]]


# -------------------------------
# MAIN ENTRY
# -------------------------------
def main():
    print("🔍 Starting EN-VI Boundary Anchor Alignment...")
    input_dir = RAW_DIR
    output_dir = RAW_UNMATCHED_ONLY_DIR

    UNMATCHED_LOG_FILE = Path(output_dir) / "unmatched_log.txt"
    FINISHED_LOG_FILE = Path(output_dir) / "finished_log.txt"
    UNMATCHED_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(UNMATCHED_LOG_FILE, "w", encoding="utf-8") as f:
        f.write("UNMATCHED / MISMATCHED CHAPTERS LOG\n\n")
    with open(FINISHED_LOG_FILE, "w", encoding="utf-8") as f:
        f.write("Last finished manga index: -1\n")

    manga_list = list(enumerate(get_all_manga(input_dir)))
    manga_list = filter_by_range(manga_list, INPUT_RANGE)

    print(f"📋 Processing {len(manga_list)} total items based on range criteria.")

    progress_bar = tqdm(manga_list, desc="Processing chapters")
    for manga_index, (manga, chapter) in progress_bar:
        progress_bar.set_description(f"Processing {manga}/{chapter}")
        try:
            with open(UNMATCHED_LOG_FILE, "a", encoding="utf-8") as log_f:
                process_chapter(
                    input_dir,
                    output_dir,
                    manga,
                    chapter,
                    manga_index,
                    log_f,
                )
        except Exception as e:
            tqdm.write(f"❌ Error: {manga}/{chapter} | {e}")
        else:
            with open(FINISHED_LOG_FILE, "w", encoding="utf-8") as finished_f:
                finished_f.write(f"Last finished manga index: {manga_index}\n")

    print("\n🎉 DONE!")


if __name__ == "__main__":
    main()