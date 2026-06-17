"""
run_recognition.py
------------------
Real-time face recognition.

Anti-spoof  : AntiSpoofChecker from anti_spoof.py
              → MiniFASNet (PyTorch) if .pth files present in models/
              → Motion-based MAD fallback otherwise

Mask detect : PyTorch MobileNetV2 if models/mask_detector.pt exists
              → YCrCb skin-tone heuristic otherwise

NO TensorFlow imports anywhere in this file.

Press Q to quit.
"""

import cv2
import numpy as np
import pickle
import os
import sys
import time
from datetime import datetime

import pandas as pd
import face_recognition

# ---------------------------------------------------------------------------
# Anti-spoof  (AntiSpoofChecker uses MiniFASNet .pth if available)
# ---------------------------------------------------------------------------
try:
    from anti_spoof import AntiSpoofChecker
except ModuleNotFoundError:
    try:
        from core.anti_spoof import AntiSpoofChecker
    except ModuleNotFoundError:
        # Absolute fallback — never blocks startup
        class AntiSpoofChecker:
            mode = "Disabled"
            def check(self, roi): return True, 1.0
            def reset(self): pass

_spoof_checker = AntiSpoofChecker()

# ---------------------------------------------------------------------------
# Mask detection  (PyTorch only — zero TensorFlow)
# ---------------------------------------------------------------------------
_mask_model     = None
_mask_use_cnn   = False

def _load_mask_model():
    """Load PyTorch mask_detector.pt only. Train with: python collect_mask_data.py"""
    global _mask_model, _mask_use_cnn

    pt_path = os.path.join("models", "mask_detector.pt")
    if not os.path.exists(pt_path):
        print("[MASK] No mask_detector.pt — using skin-tone heuristic.")
        print("[MASK] Train one with: python collect_mask_data.py")
        return
    try:
        import torch, torch.nn as nn
        from torchvision import models as tv_models
        m = tv_models.mobilenet_v2(weights=None)
        m.classifier[1] = nn.Linear(m.last_channel, 2)
        state = torch.load(pt_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        m.load_state_dict(state); m.eval()
        _mask_model   = m
        _mask_use_cnn = True
        print("[MASK] PyTorch mask model loaded (MobileNetV2).")
    except Exception as e:
        print(f"[MASK] Load failed: {e} — using heuristic.")

_load_mask_model()

def _mask_preprocess(roi_bgr):
    import torch
    img = cv2.resize(roi_bgr, (224, 224))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
    rgb = (rgb - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    return torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).float()

def _mask_cnn(roi_bgr):
    import torch, torch.nn.functional as F
    with torch.no_grad():
        p = F.softmax(_mask_model(_mask_preprocess(roi_bgr)), dim=1)[0]
    v = float(p[1])       # index 1 = with_mask
    return v >= 0.55, v

# Tune this value based on debug output:
#   bare face  → ratio typically 0.15 – 0.45
#   with mask  → ratio typically 0.00 – 0.08
# Set threshold between those two ranges.
MASK_SKIN_THRESHOLD = 0.12   # flag as masked if skin ratio < 12%
MASK_DEBUG          = False  # set True to print ratio every frame for tuning

def _mask_heuristic(roi_bgr):
    """YCrCb skin-tone: little skin in lower half → mask present.
    Wide Cr/Cb range covers light/dark/warm skin tones.
    Set MASK_DEBUG=True to tune MASK_SKIN_THRESHOLD for your lighting/skin tone.
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return False, 0.5
    h, w   = roi_bgr.shape[:2]
    if h < 20 or w < 20:
        return False, 0.5
    lower  = roi_bgr[h // 2:, :]
    ycrcb  = cv2.cvtColor(lower, cv2.COLOR_BGR2YCrCb)
    # Wide range: Cr 120-180, Cb 70-135 — covers light to dark skin tones
    skin   = cv2.inRange(ycrcb,
                         np.array([0,  120,  70], np.uint8),
                         np.array([255, 180, 135], np.uint8))
    ratio  = np.count_nonzero(skin) / (lower.shape[0] * lower.shape[1] + 1e-6)
    if MASK_DEBUG:
        print(f"[MASK_DEBUG] skin_ratio={ratio:.4f}  threshold={MASK_SKIN_THRESHOLD}")
    wearing = ratio < MASK_SKIN_THRESHOLD
    return wearing, float(1.0 - ratio if wearing else ratio)

def check_mask(roi_bgr):
    """Returns (wearing_mask: bool, confidence: float)."""
    if _mask_use_cnn:
        return _mask_cnn(roi_bgr)
    return _mask_heuristic(roi_bgr)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODELS_DIR           = "models"
ENCODINGS_FILE       = os.path.join(MODELS_DIR, "encodings.pickle")
SNAPSHOTS_DIR        = "snapshots"
EXPORTS_DIR          = "exports"
LOG_FILE_PATH        = os.path.join(EXPORTS_DIR, "attendance_log_temp.csv")

TOLERANCE            = 0.42          # face distance threshold (lower = stricter)
CV_SCALER            = 4             # downscale factor — 2=better accuracy, 4=faster
ATTENDANCE_COOLDOWN  = 60            # seconds between duplicate log entries
SHOW_CONFIDENCE      = False  # shown in terminal only, not on bounding box
ENABLE_ANTISPOOF     = True
ENABLE_MASK_DETECT   = True

MIN_FACE_PX          = 60            # ignore boxes smaller than this
MASK_WARN_COOLDOWN   = 10            # seconds between "remove mask" prints
UNKNOWN_COOLDOWN     = 60            # seconds between logging the same unknown face

WINDOW_W, WINDOW_H   = 1280, 720
WINDOW_NAME          = "FaceX - Recognition  [Q to quit]"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def load_encodings():
    print("[INFO] Loading face encodings...")
    if not os.path.exists(ENCODINGS_FILE):
        print(f"[ERROR] '{ENCODINGS_FILE}' not found. Run encoder.py first.")
        sys.exit(1)
    with open(ENCODINGS_FILE, "rb") as f:
        data = pickle.load(f)
    print(f"[INFO] Loaded {len(data['encodings'])} encodings "
          f"for {len(set(data['names']))} person(s).")
    return data["encodings"], data["names"]

def distance_to_confidence(dist):
    """
    dist=0.0  → 100%  (perfect match)
    dist=TOLERANCE → 0%
    """
    return float(np.clip((1.0 - dist / TOLERANCE) * 100.0, 0.0, 100.0))

_snapshot_taken = set()   # track who already has a snapshot this session

def save_snapshot(roi, name, unique=False):
    """Save snapshot. For known people: one per session (reused on repeat).
    For unknown/security events: unique=True saves a new file each time."""
    ensure_dir(SNAPSHOTS_DIR)
    if not unique and name in _snapshot_taken:
        # Known person already snapped this session — reuse existing file
        existing = [
            os.path.join(SNAPSHOTS_DIR, f)
            for f in os.listdir(SNAPSHOTS_DIR)
            if f.startswith(name + "_") and f.endswith(".jpg")
        ]
        if existing:
            return sorted(existing)[-1]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fp = os.path.join(SNAPSHOTS_DIR, f"{name}_{ts}.jpg")
    cv2.imwrite(fp, roi)
    if not unique:
        _snapshot_taken.add(name)
    return fp

# ---------------------------------------------------------------------------
# Attendance state
# ---------------------------------------------------------------------------
attendance_log    = []
last_logged_time  = {}
last_mask_warn    = {}
last_unknown_time = 0.0              # global cooldown for unknown logging

def mark_attendance(name, confidence, snapshot_path):
    now = datetime.now()
    if name in last_logged_time:
        if (now - last_logged_time[name]).total_seconds() < ATTENDANCE_COOLDOWN:
            return
    last_logged_time[name] = now
    attendance_log.append({
        "Name":       name,
        "Date":       now.strftime("%Y-%m-%d"),
        "Time":       now.strftime("%H:%M:%S"),
        "Confidence": f"{confidence:.1f}%",
        "Snapshot":   snapshot_path,
    })
    print(f"[ATTENDANCE] {name} | Conf: {confidence:.1f}% | {now.strftime('%H:%M:%S')}")

def save_log():
    if not attendance_log:
        print("[INFO] No attendance entries to save.")
        return
    try:
        os.makedirs(EXPORTS_DIR, exist_ok=True)
        pd.DataFrame(attendance_log).to_csv(LOG_FILE_PATH, index=False)
        print(f"[LOG] Attendance saved -> {LOG_FILE_PATH}")
    except Exception as e:
        print(f"[LOG ERROR] {e}")

# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
def open_camera():
    """Cross-platform camera initialization"""

    if os.name == "nt":
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    else:
        backends = [cv2.CAP_ANY]

    for backend in backends:
        for idx in range(3):
            cap = cv2.VideoCapture(idx, backend)

            if not cap.isOpened():
                continue

            time.sleep(0.7)

            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                print(f"[INFO] Camera opened (index={idx}, backend={backend})")

                cap.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_W)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_H)

                try:
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                except:
                    pass

                return cap

            cap.release()

    print("[ERROR] No working camera found.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def run_recognition():
    known_encodings, known_names = load_encodings()
    ensure_dir(SNAPSHOTS_DIR)

    cap = open_camera()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WINDOW_W, WINDOW_H)

    mask_mode  = ("CNN" if _mask_use_cnn else "Heuristic") if ENABLE_MASK_DETECT else "Off"
    print(f"[INFO] Anti-spoof: {_spoof_checker.mode} | "
          f"Mask: {mask_mode} | Q to quit.")

    frame_count = 0
    fps_start   = time.time()
    fps         = 0.0

    while True:
        
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            # On Windows, if the frame fails, we need to wait slightly and retry
            time.sleep(0.03) 
            continue

        # Ensure the frame has actual data before resizing
        if frame.shape[0] < 10 or frame.shape[1] < 10:
            continue


        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        fh, fw = frame.shape[:2]

        # ── Detect faces on downscaled frame ──────────────────────────────
        small     = cv2.resize(frame, (0, 0),
                               fx=1/CV_SCALER, fy=1/CV_SCALER,
                               interpolation=cv2.INTER_AREA)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        face_locs = face_recognition.face_locations(rgb_small, model="hog")
        face_encs = face_recognition.face_encodings(rgb_small, face_locs)

        for (t_s, r_s, b_s, l_s), enc in zip(face_locs, face_encs):
            # Scale back to full frame
            top    = max(0,  t_s * CV_SCALER)
            right  = min(fw, r_s * CV_SCALER)
            bottom = min(fh, b_s * CV_SCALER)
            left   = max(0,  l_s * CV_SCALER)

            # Skip tiny / ghost boxes
            if (right - left) < MIN_FACE_PX or (bottom - top) < MIN_FACE_PX:
                continue

            roi = frame[top:bottom, left:right].copy()
            if roi.size == 0:
                continue

            # ── Mask check ────────────────────────────────────────────────
            wearing_mask = False
            if ENABLE_MASK_DETECT:
                wearing_mask, mask_conf = check_mask(roi)
                if wearing_mask:
                    pos_key  = f"{left//80}_{top//80}"
                    now_ts   = time.time()
                    if now_ts - last_mask_warn.get(pos_key, 0) > MASK_WARN_COOLDOWN:
                        last_mask_warn[pos_key] = now_ts
                        print(f"[MASK WARNING] Person at ({left},{top}) is wearing a mask "
                              f"({mask_conf*100:.0f}% conf). Please remove mask.")

            # ── Anti-spoof check ──────────────────────────────────────────
            is_live = True
            if ENABLE_ANTISPOOF:
                is_live, _ = _spoof_checker.check(roi)

            # ── Recognition ───────────────────────────────────────────────
            label      = "Unknown"
            confidence = 0.0

            if not is_live:
                label = "SPOOF DETECTED"
                color = (0, 100, 255)

            elif wearing_mask:
                label = "Remove Mask"
                color = (0, 200, 255)

            else:
                if known_encodings:
                    distances  = face_recognition.face_distance(known_encodings, enc)
                    best_idx   = int(np.argmin(distances))
                    min_dist   = float(distances[best_idx])
                    confidence = distance_to_confidence(min_dist)

                    if min_dist < TOLERANCE:
                        label = known_names[best_idx]
                        snap  = save_snapshot(roi, label)
                        mark_attendance(label, confidence, snap)

                # ── Log unknown face for security ──────────────────────────
                if label == "Unknown":
                    global last_unknown_time
                    now_ts = time.time()
                    if now_ts - last_unknown_time >= UNKNOWN_COOLDOWN:
                        last_unknown_time = now_ts
                        snap = save_snapshot(roi, "Unknown", unique=True)
                        now  = datetime.now()
                        attendance_log.append({
                            "Name":       "Unknown",
                            "Date":       now.strftime("%Y-%m-%d"),
                            "Time":       now.strftime("%H:%M:%S"),
                            "Confidence": "N/A",
                            "Snapshot":   snap,
                        })
                        print(f"[SECURITY] Unknown face detected & logged | {now.strftime('%H:%M:%S')}")

                color = (0, 0, 210) if label == "Unknown" else (0, 210, 0)

            # ── Draw ──────────────────────────────────────────────────────
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            bar_top = max(top - 36, 0)
            cv2.rectangle(frame, (left, bar_top), (right, top), color, cv2.FILLED)

            if SHOW_CONFIDENCE and label not in ("Unknown", "SPOOF DETECTED",
                                                  "Remove Mask"):
                disp = f"{label}  {confidence:.0f}%"
            else:
                disp = label

            cv2.putText(frame, disp, (left + 4, top - 8),
                        cv2.FONT_HERSHEY_DUPLEX, 0.78, (255, 255, 255), 1)

        # ── FPS overlay ───────────────────────────────────────────────────
        frame_count += 1
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            fps         = frame_count / elapsed
            frame_count = 0
            fps_start   = time.time()

        cv2.putText(frame, f"FPS: {fps:.1f}",
                    (fw - 150, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break

    cap.release()
    cv2.destroyAllWindows()
    save_log()
    print("[INFO] Recognition session ended.")


if __name__ == "__main__":
    run_recognition()
