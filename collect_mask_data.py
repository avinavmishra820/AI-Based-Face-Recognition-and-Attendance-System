"""
collect_mask_data.py
--------------------
Step 1: Collect face images WITH mask and WITHOUT mask from webcam.
Step 2: Auto-trains mask_detector.pt when done.

Usage:
    python collect_mask_data.py

Controls during capture:
    SPACE  — save current frame as a sample
    ENTER  — done with current phase, move to next
    Q      — quit early
"""
import os, sys, time, cv2, subprocess
import numpy as np

WITH_DIR    = os.path.join("dataset", "mask_train", "with_mask")
WITHOUT_DIR = os.path.join("dataset", "mask_train", "without_mask")
TARGET      = 120   # images per class (enough for good accuracy)
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

os.makedirs(WITH_DIR,    exist_ok=True)
os.makedirs(WITHOUT_DIR, exist_ok=True)


def count_existing(d):
    return len([f for f in os.listdir(d) if f.endswith(".jpg")])


def collect_phase(label, save_dir, color):
    """Collect face crops for one phase. Returns number saved."""
    existing = count_existing(save_dir)
    needed   = max(0, TARGET - existing)

    if needed == 0:
        print(f"[{label}] Already have {existing} images — skipping capture.")
        return existing

    print(f"\n{'='*55}")
    print(f"  PHASE: {label}")
    print(f"  Saved so far : {existing} | Need {needed} more (target {TARGET})")
    if label == "WITH MASK":
        print("  → PUT YOUR MASK ON NOW")
    else:
        print("  → MAKE SURE YOUR FACE IS BARE (no mask)")
    print(f"  SPACE = save face | ENTER = finish | Q = quit")
    print(f"{'='*55}")
    input("  Press ENTER when ready to start camera...")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    saved  = existing
    idx    = existing
    last_t = 0

    while True:
        ret, frame = cap.read()
        if not ret: continue

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

        disp = frame.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(disp, (x, y), (x+w, y+h), color, 2)

        # HUD
        done_pct = int((saved / TARGET) * 100)
        bar      = "█" * (done_pct // 5) + "░" * (20 - done_pct // 5)
        cv2.putText(disp, f"{label}  [{bar}] {saved}/{TARGET}",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        cv2.putText(disp, "SPACE=save  ENTER=done  Q=quit",
                    (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        if saved >= TARGET:
            cv2.putText(disp, "Target reached! Press ENTER to continue.",
                        (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Mask Data Collector", disp)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            cap.release(); cv2.destroyAllWindows(); sys.exit(0)

        if key == 13:  # ENTER
            break

        if key == ord(' ') and len(faces) > 0:
            now = time.time()
            if now - last_t < 0.15:   # debounce
                continue
            last_t = now
            x, y, w, h = faces[0]
            # Add small padding
            pad  = int(0.1 * min(w, h))
            fh, fw = frame.shape[:2]
            x1 = max(0, x - pad);  y1 = max(0, y - pad)
            x2 = min(fw, x+w+pad); y2 = min(fh, y+h+pad)
            roi  = frame[y1:y2, x1:x2]
            fname = os.path.join(save_dir, f"{label.replace(' ','_')}_{idx:04d}.jpg")
            cv2.imwrite(fname, roi)
            idx   += 1
            saved += 1
            print(f"  Saved {saved}/{TARGET}: {os.path.basename(fname)}")

        # Auto-save every 0.5s if space held (continuous mode)
        if saved >= TARGET + 20:   # safety stop
            break

    cap.release()
    cv2.destroyAllWindows()
    return saved


def main():
    print("\n╔══════════════════════════════════════════════════╗")
    print("║        FaceX — Mask CNN Data Collector           ║")
    print("╚══════════════════════════════════════════════════╝")
    print(f"  Saving to:")
    print(f"    With mask    → {WITH_DIR}")
    print(f"    Without mask → {WITHOUT_DIR}")
    print(f"  Target: {TARGET} images per class\n")

    # Check existing
    wm  = count_existing(WITH_DIR)
    wom = count_existing(WITHOUT_DIR)
    print(f"  Existing — with mask: {wm}  |  without mask: {wom}")

    # Phase 1: without mask
    collect_phase("WITHOUT MASK", WITHOUT_DIR, (0, 200, 0))

    # Phase 2: with mask
    collect_phase("WITH MASK",    WITH_DIR,    (0, 120, 255))

    # Summary
    wm  = count_existing(WITH_DIR)
    wom = count_existing(WITHOUT_DIR)
    print(f"\n  Collection done — with: {wm}  without: {wom}")

    if wm < 20 or wom < 20:
        print("[ERROR] Need at least 20 images per class to train.")
        sys.exit(1)

    # Train
    print("\n  Starting training...\n")
    cmd = [
        sys.executable, "mask_detection.py",
        "--train",
        "--with-mask",    WITH_DIR,
        "--without-mask", WITHOUT_DIR,
        "--epochs", "20"
    ]
    subprocess.run(cmd, check=True)
    print("\n  ✓ mask_detector.pt saved to models/")
    print("  Run python run_recognition.py — CNN mask detection is now active.")


if __name__ == "__main__":
    main()
