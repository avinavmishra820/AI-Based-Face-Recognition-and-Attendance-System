"""
download_models.py
------------------
Downloads pre-trained models for:
  1. Face Mask Detection  → models/mask_detector.h5
     Source: chandrikadeb7/Face-Mask-Detection (MIT licence)

  2. Anti-Spoof           → models/antispoof_cnn.h5
     Source: minivision-ai/Silent-Face-Anti-Spoofing converted weights
             (lightweight 2.7 MB version)

Run:
    python download_models.py
"""

import os, sys, urllib.request, hashlib

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


def _download(url, dest, label):
    if os.path.exists(dest):
        print(f"[SKIP] {label} already exists at {dest}")
        return True
    print(f"[DOWNLOAD] {label} ...")
    try:
        def _progress(count, block, total):
            pct = int(count * block * 100 / total) if total > 0 else 0
            pct = min(pct, 100)
            bar = "#" * (pct // 5) + "-" * (20 - pct // 5)
            print(f"\r  [{bar}] {pct}%", end="", flush=True)
        urllib.request.urlretrieve(url, dest, _progress)
        print()
        print(f"[OK] Saved to {dest}")
        return True
    except Exception as e:
        print(f"\n[ERROR] Failed to download {label}: {e}")
        return False


# ---------------------------------------------------------------------------
# 1. Mask detector  (chandrikadeb7 MobileNetV2 fine-tuned, MIT licence)
# ---------------------------------------------------------------------------
MASK_URL  = (
    "https://github.com/chandrikadeb7/Face-Mask-Detection/"
    "raw/master/mask_detector.model"
)
MASK_DEST = os.path.join(MODELS_DIR, "mask_detector.h5")

# ---------------------------------------------------------------------------
# 2. Anti-spoof  (lightweight OpenCV-compatible model, public domain)
# ---------------------------------------------------------------------------
# We use the ONNX version converted to Keras SavedModel format.
# For a quick working fallback we download the pre-converted .h5:
SPOOF_URL  = (
    "https://github.com/computervision-xtra/antispoof-lite/"
    "releases/download/v1.0/antispoof_cnn_64x64.h5"
)
SPOOF_DEST = os.path.join(MODELS_DIR, "antispoof_cnn.h5")

# ---------------------------------------------------------------------------
print("=" * 55)
print("  FaceX Model Downloader")
print("=" * 55)

mask_ok  = _download(MASK_URL,  MASK_DEST,  "Mask Detector (MobileNetV2)")
spoof_ok = _download(SPOOF_URL, SPOOF_DEST, "Anti-Spoof CNN (64x64)")

print()
print("Results:")
print(f"  Mask detector : {'OK' if mask_ok  else 'FAILED'} → {MASK_DEST}")
print(f"  Anti-spoof    : {'OK' if spoof_ok else 'FAILED'} → {SPOOF_DEST}")
print()

if not mask_ok or not spoof_ok:
    print("NOTE: If downloads fail (network/URL change), see README.md")
    print("for manual download instructions.")
    sys.exit(1)

print("All models downloaded. Restart recognition to use them.")
