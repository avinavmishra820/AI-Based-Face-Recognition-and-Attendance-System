"""
anti_spoof.py — MiniFASNet with exact checkpoint key matching.

Verified from dump_se_keys.py output:

SE block keys (named attributes — no Sequential):
  .se_fc1.weight   (reduction, out_c, 1, 1)
  .se_bn1.*        BN on reduced channels
  .se_fc2.weight   (out_c, reduction, 1, 1)
  .se_bn2.*        BN on out_c channels

SE blocks present in V1SE only, at exactly:
  conv_3.model.3   → channels=64, reduction=16
  conv_4.model.5   → channels=128, reduction=32
  conv_5.model.1   → channels=128, reduction=32

V1SE conv_4 groups (from size mismatch):
  model.0: 154,  model.1: 52,  model.2: 26,
  model.3: 52,   model.4: 26,  model.5: 26

Both models: linear(512→128, bias=False)  ← ckpt has NO linear.bias key
"""

import os
import cv2
import numpy as np
from collections import deque

MODELS_DIR = "models"


def _make_arch():
    try:
        import torch
        import torch.nn as nn

        # ── conv + bn + prelu ──────────────────────────────────────────────
        class CBP(nn.Module):
            def __init__(self, in_c, out_c, k=3, s=1, p=1, g=1):
                super().__init__()
                self.conv  = nn.Conv2d(in_c, out_c, k, s, p, groups=g, bias=False)
                self.bn    = nn.BatchNorm2d(out_c)
                self.prelu = nn.PReLU(out_c)
            def forward(self, x):
                return self.prelu(self.bn(self.conv(x)))

        # ── project: named .conv + .bn (NOT nn.Sequential) ────────────────
        class Project(nn.Module):
            def __init__(self, in_c, out_c):
                super().__init__()
                self.conv = nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False)
                self.bn   = nn.BatchNorm2d(out_c)
            def forward(self, x):
                return self.bn(self.conv(x))

        # ── conv_6_dw: named .conv + .bn (NOT nn.Sequential) ──────────────
        class DWHead(nn.Module):
            def __init__(self, c, k=5):
                super().__init__()
                self.conv = nn.Conv2d(c, c, k, 1, k // 2, groups=c, bias=False)
                self.bn   = nn.BatchNorm2d(c)
            def forward(self, x):
                return self.bn(self.conv(x))

        # ── SE block: keys se_fc1, se_bn1, se_fc2, se_bn2 ─────────────────
        # Confirmed key names from dump_se_keys.py:
        #   .se_fc1.weight  (reduction, out_c, 1, 1)
        #   .se_bn1.*       BN(reduction)
        #   .se_fc2.weight  (out_c, reduction, 1, 1)
        #   .se_bn2.*       BN(out_c)
        class SE(nn.Module):
            def __init__(self, channels, reduction):
                super().__init__()
                self.se_fc1 = nn.Conv2d(channels, reduction, 1, bias=False)
                self.se_bn1 = nn.BatchNorm2d(reduction)
                self.se_fc2 = nn.Conv2d(reduction, channels, 1, bias=False)
                self.se_bn2 = nn.BatchNorm2d(channels)

            def forward(self, x):
                s = x.mean(dim=(2, 3), keepdim=True)        # global avg pool
                s = torch.relu(self.se_bn1(self.se_fc1(s)))
                s = torch.sigmoid(self.se_bn2(self.se_fc2(s)))
                return x * s

        # ── Standard DepthWise block ───────────────────────────────────────
        class DW(nn.Module):
            def __init__(self, in_c, out_c, g, s=1):
                super().__init__()
                self.conv    = CBP(in_c, g, 1, 1, 0)
                self.conv_dw = CBP(g, g, 3, s, 1, g=g)
                self.project = Project(g, out_c)
            def forward(self, x):
                return self.project(self.conv_dw(self.conv(x)))

        # ── DepthWise + residual (no SE) ───────────────────────────────────
        class DWR(nn.Module):
            def __init__(self, in_c, out_c, g, s=1):
                super().__init__()
                self.use_res = (s == 1 and in_c == out_c)
                self.conv    = CBP(in_c, g, 1, 1, 0)
                self.conv_dw = CBP(g, g, 3, s, 1, g=g)
                self.project = Project(g, out_c)
            def forward(self, x):
                out = self.project(self.conv_dw(self.conv(x)))
                return out + x if self.use_res else out

        # ── DepthWise + residual + SE ──────────────────────────────────────
        # Used at conv_3.model.3, conv_4.model.5, conv_5.model.1 in V1SE
        class DWRSE(nn.Module):
            def __init__(self, in_c, out_c, g, reduction, s=1):
                super().__init__()
                self.use_res = (s == 1 and in_c == out_c)
                self.conv    = CBP(in_c, g, 1, 1, 0)
                self.conv_dw = CBP(g, g, 3, s, 1, g=g)
                self.project = Project(g, out_c)
                # SE keys must be named se_fc1/se_bn1/se_fc2/se_bn2
                # (stored directly on this module, not nested)
                se = SE(out_c, reduction)
                self.se_fc1 = se.se_fc1
                self.se_bn1 = se.se_bn1
                self.se_fc2 = se.se_fc2
                self.se_bn2 = se.se_bn2

            def _se(self, x):
                s = x.mean(dim=(2, 3), keepdim=True)
                s = torch.relu(self.se_bn1(self.se_fc1(s)))
                s = torch.sigmoid(self.se_bn2(self.se_fc2(s)))
                return x * s

            def forward(self, x):
                out = self.project(self.conv_dw(self.conv(x)))
                out = self._se(out)
                return out + x if self.use_res else out

        # ── ResStack (ModuleList → keys .model.0.* .model.1.* ...) ────────
        class ResStack(nn.Module):
            def __init__(self, blocks):
                super().__init__()
                self.model = nn.ModuleList(blocks)
            def forward(self, x):
                for b in self.model:
                    x = b(x)
                return x

        # ── MiniFASNetV2  (2.7_80x80) — no SE blocks ──────────────────────
        class MiniFASNetV2(nn.Module):
            def __init__(self, num_classes=3):
                super().__init__()
                self.conv1      = CBP(3,   32, 3, 1, 1)
                self.conv2_dw   = CBP(32,  32, 3, 1, 1, g=32)
                self.conv_23    = DW (32,  64,  g=103, s=2)
                self.conv_3     = ResStack([
                    DWR(64,  64,  g=13),
                    DWR(64,  64,  g=13),
                    DWR(64,  64,  g=13),
                    DWR(64,  64,  g=13),
                ])
                self.conv_34    = DW (64,  128, g=231, s=2)
                self.conv_4     = ResStack([
                    DWR(128, 128, g=231),
                    DWR(128, 128, g=52),
                    DWR(128, 128, g=26),
                    DWR(128, 128, g=77),
                    DWR(128, 128, g=26),
                    DWR(128, 128, g=26),
                ])
                self.conv_45    = DW (128, 128, g=308, s=1)
                self.conv_5     = ResStack([
                    DWR(128, 128, g=26),
                    DWR(128, 128, g=26),
                ])
                self.conv_6_sep = CBP(128, 512, 1, 1, 0)
                self.conv_6_dw  = DWHead(512, k=5)
                self.linear     = nn.Linear(512, 128, bias=False)
                self.bn         = nn.BatchNorm1d(128)
                self.prob       = nn.Linear(128, num_classes, bias=False)

            def forward(self, x):
                x = self.conv1(x);      x = self.conv2_dw(x)
                x = self.conv_23(x);    x = self.conv_3(x)
                x = self.conv_34(x);    x = self.conv_4(x)
                x = self.conv_45(x);    x = self.conv_5(x)
                x = self.conv_6_sep(x); x = self.conv_6_dw(x)
                x = nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
                return self.prob(self.bn(self.linear(x)))

        # ── MiniFASNetV1SE  (4_0_0_80x80) — SE at model.3/5/1 ────────────
        # SE positions confirmed from dump_se_keys.py:
        #   conv_3.model.3  → DWRSE(64→64,  g=52, reduction=16)
        #   conv_4.model.5  → DWRSE(128→128,g=26, reduction=32)
        #   conv_5.model.1  → DWRSE(128→128,g=26, reduction=32)
        # conv_4 groups confirmed from size mismatch error:
        #   model.0:154  model.1:52  model.2:26  model.3:52  model.4:26  model.5:26
        class MiniFASNetV1SE(nn.Module):
            def __init__(self, num_classes=3):
                super().__init__()
                self.conv1      = CBP(3,   32, 3, 1, 1)
                self.conv2_dw   = CBP(32,  32, 3, 1, 1, g=32)
                self.conv_23    = DW (32,  64,  g=103, s=2)
                self.conv_3     = ResStack([
                    DWR (64,  64,  g=13),
                    DWR (64,  64,  g=26),
                    DWR (64,  64,  g=13),
                    DWRSE(64, 64,  g=52, reduction=16),   # ← SE here
                ])
                self.conv_34    = DW (64,  128, g=231, s=2)
                self.conv_4     = ResStack([
                    DWR (128, 128, g=154),
                    DWR (128, 128, g=52),
                    DWR (128, 128, g=26),
                    DWR (128, 128, g=52),
                    DWR (128, 128, g=26),
                    DWRSE(128, 128, g=26, reduction=32),  # ← SE here
                ])
                self.conv_45    = DW (128, 128, g=308, s=1)
                self.conv_5     = ResStack([
                    DWR (128, 128, g=26),
                    DWRSE(128, 128, g=26, reduction=32),  # ← SE here
                ])
                self.conv_6_sep = CBP(128, 512, 1, 1, 0)
                self.conv_6_dw  = DWHead(512, k=5)
                self.linear     = nn.Linear(512, 128, bias=False)
                self.bn         = nn.BatchNorm1d(128)
                self.prob       = nn.Linear(128, num_classes, bias=False)

            def forward(self, x):
                x = self.conv1(x);      x = self.conv2_dw(x)
                x = self.conv_23(x);    x = self.conv_3(x)
                x = self.conv_34(x);    x = self.conv_4(x)
                x = self.conv_45(x);    x = self.conv_5(x)
                x = self.conv_6_sep(x); x = self.conv_6_dw(x)
                x = nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
                return self.prob(self.bn(self.linear(x)))

        return {"MiniFASNetV2": MiniFASNetV2, "MiniFASNetV1SE": MiniFASNetV1SE}

    except ImportError:
        return {}


# ---------------------------------------------------------------------------
class _MiniFASLoader:
    REAL_IDX = 2   # confirmed: 0=spoof_type1, 1=spoof_type2, 2=real/live

    def __init__(self):
        self._models    = []
        self._available = False
        self._try_load()

    def _try_load(self):
        arch = _make_arch()
        if not arch:
            print("[AntiSpoof] PyTorch not available — motion fallback.")
            return

        import torch

        if not os.path.isdir(MODELS_DIR):
            return

        pth_files = sorted([
            f for f in os.listdir(MODELS_DIR)
            if f.endswith(".pth") and "MiniFAS" in f
        ])
        if not pth_files:
            print("[AntiSpoof] No MiniFASNet .pth files — motion fallback.")
            return

        for fname in pth_files:
            fpath = os.path.join(MODELS_DIR, fname)
            cls   = arch["MiniFASNetV1SE"] if "V1SE" in fname else arch["MiniFASNetV2"]
            try:
                model = cls(num_classes=3)
                raw   = torch.load(fpath, map_location="cpu", weights_only=False)
                sd    = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
                sd    = {k.replace("module.", ""): v for k, v in sd.items()}

                missing, unexpected = model.load_state_dict(sd, strict=False)
                if missing:
                    print(f"[AntiSpoof] {fname}: {len(missing)} missing — {missing[:2]}")
                else:
                    print(f"[AntiSpoof] {fname}: 0 missing keys ✓")

                model.eval()
                self._models.append(model)
                print(f"[AntiSpoof] Loaded: {fname}")

            except Exception as e:
                print(f"[AntiSpoof] Failed to load {fname}: {e}")

        self._available = len(self._models) > 0
        print(f"[AntiSpoof] MiniFASNet: "
              f"{'active — ' + str(len(self._models)) + ' model(s).' if self._available else 'FAILED — motion fallback.'}")

    @property
    def available(self): return self._available

    # Threshold: score must be >= this to be considered REAL.
    # Lower = stricter (more spoof detections). 0.70 recommended.
    LIVE_THRESHOLD = 0.70

    def predict(self, face_roi_bgr):
        if not self._available or face_roi_bgr is None or face_roi_bgr.size == 0:
            return True, 1.0
        try:
            import torch, torch.nn.functional as F

            # MiniFASNet requires the face to be tightly cropped —
            # run inference at 3 scales and take the minimum score (strictest).
            # This prevents a phone screen passing just because one crop looks real.
            h, w = face_roi_bgr.shape[:2]
            crops = []

            # Scale 1: full ROI as given
            crops.append(cv2.resize(face_roi_bgr, (80, 80)))

            # Scale 2: centre 80% crop (removes background noise)
            m = int(min(h, w) * 0.10)
            crops.append(cv2.resize(face_roi_bgr[m:h-m, m:w-m], (80, 80)))

            # Scale 3: upper-centre crop (forehead+eyes region — most discriminative)
            th = int(h * 0.65)
            tw = int(w * 0.80)
            tx = (w - tw) // 2
            crops.append(cv2.resize(face_roi_bgr[0:th, tx:tx+tw], (80, 80)))

            all_scores = []
            with torch.no_grad():
                for crop in crops:
                    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
                    rgb = (rgb - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
                    t   = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).float()
                    for m in self._models:
                        all_scores.append(F.softmax(m(t), dim=1)[0][self.REAL_IDX].item())

            # Use MINIMUM score across all crops+models — if any crop looks like spoof, reject
            score = float(min(all_scores))
            return score >= self.LIVE_THRESHOLD, score
        except Exception as e:
            print(f"[AntiSpoof] Inference error: {e}")
            return True, 1.0


# ---------------------------------------------------------------------------
class MotionLivenessChecker:
    def __init__(self, history_len=15, motion_threshold=2.5, min_live_frames=5):
        self.history_len=history_len; self.motion_threshold=motion_threshold
        self.min_live_frames=min_live_frames; self._prev_gray=None
        self._scores=deque(maxlen=history_len)

    def check(self, roi):
        if roi is None or roi.size == 0: return True, 0.0
        small = cv2.resize(roi, (64, 64))
        gray  = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        score = float(np.mean(cv2.absdiff(gray, self._prev_gray))) \
                if self._prev_gray is not None else 0.0
        self._scores.append(score); self._prev_gray = gray
        if len(self._scores) < 4: return True, score
        live = sum(1 for s in self._scores if s >= self.motion_threshold)
        return live >= self.min_live_frames, score

    def reset(self): self._prev_gray = None; self._scores.clear()


# ---------------------------------------------------------------------------
class AntiSpoofChecker:
    def __init__(self):
        self._fas     = _MiniFASLoader()
        self._motion  = MotionLivenessChecker()
        self._use_fas = self._fas.available

    @property
    def mode(self): return "MiniFASNet" if self._use_fas else "Motion"

    def check(self, roi):
        return self._fas.predict(roi) if self._use_fas else self._motion.check(roi)

    def reset(self): self._motion.reset()
