"""
Single-file test for Chrome Lens OCR (Vietnamese).
Select an image → print extracted text.
"""

import asyncio
import tkinter as tk
from tkinter import filedialog
from chrome_lens_py import LensAPI


def select_image():
    """Open file picker."""
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[
            ("Image files", "*.png *.jpg *.jpeg *.bmp *.webp"),
            ("All files", "*.*"),
        ],
    )
    return file_path


async def run_ocr(image_path):
    """Run OCR using Chrome Lens API."""
    api = LensAPI()

    result = await api.process_image(
        image_path=image_path, ocr_language="vi"  # Vietnamese
    )

    return result.get("ocr_text", "")


def main():
    print("Select an image...")
    image_path = select_image()

    if not image_path:
        print("No image selected.")
        return

    print(f"\nProcessing: {image_path}\n")

    text = asyncio.run(run_ocr(image_path))

    print("===== OCR RESULT =====")
    if text.strip():
        text = " ".join(text.split())
        print(text)
    else:
        print("(No text detected)")


if __name__ == "__main__":
    main()
