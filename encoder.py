"""
encoder.py
----------
Scans the ``dataset/`` folder, computes 128-d face encodings for every image,
and serialises them to ``models/encodings.pickle``.

Run:
    python encoder.py
"""

import os
import sys
import pickle
import cv2
from imutils import paths
import face_recognition

# --- Configuration ---
DATASET_DIR    = "dataset"
MODELS_DIR     = "models"
ENCODINGS_FILE = os.path.join(MODELS_DIR, "encodings.pickle")

# Sub-folders inside dataset/ that should never be encoded
# (mask training data, preview images, etc.)
SKIP_DIRS = {"mask_train", "PREVIEW"}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def encode_faces() -> None:
    ensure_dir(MODELS_DIR)

    print("[INFO] Scanning dataset folder for images…")
    all_paths   = list(paths.list_images(DATASET_DIR))

    # Exclude sub-folders that are not person identity folders
    # (e.g. mask_train/with_mask, mask_train/without_mask, PREVIEW)
    image_paths = [
        p for p in all_paths
        if not any(skip in p.split(os.path.sep) for skip in SKIP_DIRS)
    ]

    skipped_dirs = len(all_paths) - len(image_paths)
    if skipped_dirs:
        print(f"[INFO] Skipped {skipped_dirs} image(s) from excluded folders: {SKIP_DIRS}")

    if not image_paths:
        print(f"[ERROR] No images found in '{DATASET_DIR}'. Run image_capture.py first.")
        sys.exit(1)

    known_encodings: list = []
    known_names:     list = []
    skipped          = 0
    total            = len(image_paths)
    BAR_WIDTH        = 30

    for i, image_path in enumerate(image_paths, start=1):
        # Derive person name from immediate parent folder
        name = image_path.split(os.path.sep)[-2]

        # ── Progress bar ────────────────────────────────────────────────
        pct      = i / total
        filled   = int(BAR_WIDTH * pct)
        bar      = "█" * filled + "░" * (BAR_WIDTH - filled)
        status   = f"\r[{bar}] {i}/{total} ({pct*100:.1f}%)  Encoding: {name:<20}"
        print(status, end="", flush=True)

        image = cv2.imread(image_path)
        if image is None:
            print(f"\n[WARNING] Cannot read '{image_path}'. Skipping.")
            skipped += 1
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect face bounding boxes (HOG for CPU-friendly speed)
        boxes = face_recognition.face_locations(rgb, model="hog")

        if not boxes:
            print(f"\n[WARNING] No face detected in '{image_path}'. Skipping.")
            skipped += 1
            continue

        # Compute 128-d encodings for each face found
        encodings = face_recognition.face_encodings(rgb, boxes)
        for encoding in encodings:
            known_encodings.append(encoding)
            known_names.append(name)

    # Move to new line after progress bar finishes
    print()

    if not known_encodings:
        print("[ERROR] No valid face encodings generated. Check your dataset images.")
        sys.exit(1)

    print(f"[INFO] Encoding complete — {len(known_encodings)} encoding(s) from "
          f"{total - skipped}/{total} image(s)  ({skipped} skipped)")
    print("[INFO] Saving encodings…")

    data = {"encodings": known_encodings, "names": known_names}
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(data, f)

    print(f"[INFO] Done. Encodings saved to '{ENCODINGS_FILE}'.")


if __name__ == "__main__":
    encode_faces()
