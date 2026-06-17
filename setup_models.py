"""
setup_models.py
---------------
Downloads verified pretrained models for FaceX:

1. MASK DETECTOR  → models/mask_detector.h5
   Source: chandrikadeb7/Face-Mask-Detection (MIT licence)
   Method: git clone (uses git-lfs for the .model file)

2. ANTI-SPOOF     → models/antispoof.onnx
   Source: minivision-ai/Silent-Face-Anti-Spoofing (Apache-2.0)
   Method: git clone then copy .onnx files

   NOTE: The anti-spoof module in run_recognition.py uses ONNX Runtime
   (pip install onnxruntime) — no TensorFlow needed for this model.

Run:
    python setup_models.py
"""

import os
import sys
import shutil
import subprocess

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
def _run(cmd, cwd=None):
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [STDERR] {result.stderr.strip()}")
    return result.returncode == 0


def _check_tool(name):
    return shutil.which(name) is not None


# ---------------------------------------------------------------------------
# 1. Mask detector
# ---------------------------------------------------------------------------
def download_mask_model():
    dest = os.path.join(MODELS_DIR, "mask_detector.h5")
    if os.path.exists(dest) and os.path.getsize(dest) > 100_000:
        print(f"[SKIP] Mask model already at {dest}")
        return True

    print("\n[1/2] Downloading Face Mask Detector (chandrikadeb7/Face-Mask-Detection)...")

    if not _check_tool("git"):
        print("[ERROR] git is not installed. Install it with: sudo dnf install git")
        return False

    tmp = "/tmp/face_mask_detection_clone"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)

    # Shallow clone — much faster than full history
    ok = _run(["git", "clone", "--depth", "1",
               "https://github.com/chandrikadeb7/Face-Mask-Detection.git",
               tmp])
    if not ok:
        print("[ERROR] git clone failed.")
        return False

    # The model file is tracked by git-lfs; check if it's real
    model_src = os.path.join(tmp, "mask_detector.model")
    if os.path.exists(model_src) and os.path.getsize(model_src) > 100_000:
        shutil.copy(model_src, dest)
        print(f"[OK] Mask model saved → {dest}")
        shutil.rmtree(tmp)
        return True
    else:
        # git-lfs not installed — try installing it
        print("[INFO] git-lfs not found or model not pulled. Trying git lfs pull...")
        if _check_tool("git-lfs"):
            _run(["git", "lfs", "pull"], cwd=tmp)
            if os.path.exists(model_src) and os.path.getsize(model_src) > 100_000:
                shutil.copy(model_src, dest)
                print(f"[OK] Mask model saved → {dest}")
                shutil.rmtree(tmp)
                return True
        print("[WARN] Could not get mask_detector.model via git-lfs.")
        print("       Install git-lfs:  sudo dnf install git-lfs && git lfs install")
        print("       Then rerun this script.")
        shutil.rmtree(tmp, ignore_errors=True)
        return False


# ---------------------------------------------------------------------------
# 2. Anti-spoof ONNX models (minivision — no TensorFlow needed)
# ---------------------------------------------------------------------------
def download_antispoof_onnx():
    dest1 = os.path.join(MODELS_DIR, "antispoof_2_7.onnx")
    dest2 = os.path.join(MODELS_DIR, "antispoof_4.onnx")

    if (os.path.exists(dest1) and os.path.getsize(dest1) > 100_000 and
            os.path.exists(dest2) and os.path.getsize(dest2) > 100_000):
        print(f"[SKIP] Anti-spoof ONNX models already in {MODELS_DIR}/")
        return True

    print("\n[2/2] Downloading Anti-Spoof ONNX models (minivision-ai)...")

    if not _check_tool("git"):
        print("[ERROR] git is not installed.")
        return False

    tmp = "/tmp/silent_face_clone"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)

    ok = _run(["git", "clone", "--depth", "1",
               "https://github.com/minivision-ai/Silent-Face-Anti-Spoofing.git",
               tmp])
    if not ok:
        print("[ERROR] git clone failed.")
        return False

    # Copy both model files (they are plain .pth files, not lfs)
    src_dir = os.path.join(tmp, "resources", "anti_spoof_models")
    found = False
    if os.path.isdir(src_dir):
        for fname in os.listdir(src_dir):
            src = os.path.join(src_dir, fname)
            dst = os.path.join(MODELS_DIR, fname)
            if os.path.getsize(src) > 10_000:
                shutil.copy(src, dst)
                print(f"  Copied: {fname} → {MODELS_DIR}/")
                found = True
    
    shutil.rmtree(tmp, ignore_errors=True)

    if found:
        print(f"[OK] Anti-spoof models saved to {MODELS_DIR}/")
        print("[NOTE] These are .pth (PyTorch) models used via the minivision inference code.")
        print("       Install: pip install onnxruntime torch torchvision")
        return True
    else:
        print("[WARN] No model files found in resources/anti_spoof_models/")
        return False


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  FaceX — Model Setup")
    print("=" * 60)

    mask_ok   = download_mask_model()
    spoof_ok  = download_antispoof_onnx()

    print("\n" + "=" * 60)
    print("  Results:")
    print(f"  Mask detector  : {'✓ OK' if mask_ok  else '✗ FAILED'}")
    print(f"  Anti-spoof     : {'✓ OK' if spoof_ok else '✗ FAILED'}")
    print("=" * 60)

    if not mask_ok or not spoof_ok:
        print("""
If downloads fail, install manually:

  # git-lfs (for mask model):
  sudo dnf install git-lfs
  git lfs install
  python setup_models.py

  # Anti-spoof via pip (alternative, no model file needed):
  pip install onnxruntime
  # The motion-based fallback in anti_spoof.py will still work without a model.
""")
        sys.exit(1)

    print("\nAll done. Restart gui_dashboard.py or run_recognition.py to use the models.")
