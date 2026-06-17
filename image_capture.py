"""
image_capture.py
----------------
Captures face images from webcam and saves them to the dataset folder.

Usage:
    python image_capture.py                  # Interactive mode (prompts for name)
    python image_capture.py "John Doe" AUTO  # Auto-capture ~150 images
    python image_capture.py "John Doe" MANUAL# Manual capture (press SPACE)
    python image_capture.py PREVIEW PREVIEW  # Camera preview only
"""

import cv2
import os
import sys
import time

# --- Configuration ---
DATASET_DIR = "dataset"
AUTO_CAPTURE_COUNT = 150       # Number of images in AUTO mode
AUTO_CAPTURE_DELAY = 0.15      # Seconds between auto-captures
PREVIEW_WINDOW_TITLE = "Camera Preview — Press Q to quit"
CAPTURE_WINDOW_TITLE  = "Face Capture — SPACE=Save | Q=Quit"


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def capture_faces(person_name: str, mode: str = "MANUAL") -> None:
    """
    Open webcam and capture images for *person_name*.

    Parameters
    ----------
    person_name : str
        Name used as the sub-folder inside ``dataset/``.
    mode : str
        ``"MANUAL"``  — press SPACE to save each image.
        ``"AUTO"``    — automatically saves ~150 images, then exits.
        ``"PREVIEW"`` — camera preview only, no saving.
    """
    is_preview = person_name.upper() == "PREVIEW" or mode.upper() == "PREVIEW"

    if not is_preview:
        save_dir = os.path.join(DATASET_DIR, person_name)
        ensure_dir(save_dir)
        print(f"[INFO] Images will be saved to: {save_dir}")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        return

    img_count   = 0
    last_capture = 0.0
    window_title = PREVIEW_WINDOW_TITLE if is_preview else CAPTURE_WINDOW_TITLE

    print(f"[INFO] Camera opened. Mode: {mode.upper()}")
    if mode.upper() == "MANUAL":
        print("[INFO] Press SPACE to capture. Press Q to quit.")
    elif mode.upper() == "AUTO":
        print(f"[INFO] Auto-capturing {AUTO_CAPTURE_COUNT} images. Press Q to cancel.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame.")
            break

        display = frame.copy()

        if not is_preview:
            cv2.putText(display, f"Name: {person_name}  Mode: {mode.upper()}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display, f"Captured: {img_count}",
                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow(window_title, display)
        key = cv2.waitKey(1) & 0xFF

        # Quit
        if key == ord('q') or key == ord('Q'):
            break

        if is_preview:
            continue

        # AUTO mode
        if mode.upper() == "AUTO":
            now = time.time()
            if now - last_capture >= AUTO_CAPTURE_DELAY:
                img_path = os.path.join(save_dir, f"{person_name}_{img_count:04d}.jpg")
                cv2.imwrite(img_path, frame)
                img_count += 1
                last_capture = now
                print(f"[CAPTURE] Auto saved image {img_count}/{AUTO_CAPTURE_COUNT}")
                if img_count >= AUTO_CAPTURE_COUNT:
                    print("[INFO] Auto-capture complete.")
                    break

        # MANUAL mode
        elif mode.upper() == "MANUAL" and key == ord(' '):
            img_path = os.path.join(save_dir, f"{person_name}_{img_count:04d}.jpg")
            cv2.imwrite(img_path, frame)
            img_count += 1
            print(f"[CAPTURE] Saved: {img_path}")

    cap.release()
    cv2.destroyAllWindows()

    if not is_preview:
        print(f"[INFO] Total images captured: {img_count}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) == 3:
        # Called from dashboard: python image_capture.py "Name" MODE
        p_name = sys.argv[1]
        p_mode = sys.argv[2]
    elif len(sys.argv) == 2:
        p_name = sys.argv[1]
        p_mode = "MANUAL"
    else:
        # Interactive
        p_name = input("Enter person name: ").strip()
        if not p_name:
            print("[ERROR] No name provided. Exiting.")
            sys.exit(1)
        p_mode = input("Mode [MANUAL / AUTO] (default=MANUAL): ").strip().upper() or "MANUAL"

    capture_faces(p_name, p_mode)
