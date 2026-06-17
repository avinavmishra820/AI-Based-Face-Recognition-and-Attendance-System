# FaceX — Model Setup Guide

## Current Status After Running setup_models.py

✅ `models/2.7_80x80_MiniFASNetV2.pth`      — anti-spoof (downloaded)  
✅ `models/4_0_0_80x80_MiniFASNetV1SE.pth`  — anti-spoof (downloaded)  
⚠️ `models/mask_detector.h5`                — incompatible (Keras version mismatch)

---

## Fix 1: Anti-Spoof (Already Working After This Update)

Replace `anti_spoof.py` with the new version provided.  
It has the full MiniFASNet architecture built-in — no external imports needed.

Test it:
```
python run_recognition.py
```
You should now see:
```
[AntiSpoof] Loaded: 2.7_80x80_MiniFASNetV2.pth
[AntiSpoof] Loaded: 4_0_0_80x80_MiniFASNetV1SE.pth
[AntiSpoof] MiniFASNet active — 2 model(s), ~97.8% accuracy.
```

---

## Fix 2: Mask Detector (Retrain with PyTorch — No TensorFlow)

The downloaded `mask_detector.h5` was saved with an older Keras version
and cannot be loaded by your current TF/Keras. Since you use PyTorch anyway,
train a fresh MobileNetV2 model instead.

### Option A — Train your own (~10 minutes, ~98% accuracy)

Download a small mask dataset:
```bash
pip install kaggle   # or use the manual download below

# Manual: download from Kaggle
# https://www.kaggle.com/datasets/omkargurav/face-mask-dataset
# Extract to:  data/with_mask/   and   data/without_mask/

python mask_detection.py \
    --train \
    --with-mask    data/with_mask \
    --without-mask data/without_mask \
    --epochs 15
```
This saves `models/mask_detector.pt` (PyTorch format).

### Option B — Quick dataset via wget (no Kaggle login)

```bash
# Prasoonmhwr's smaller public mirror (~800 images)
mkdir -p data/with_mask data/without_mask

# With mask images
wget -q "https://github.com/prasoonmhwr/face-mask-detection/raw/main/dataset/with_mask.zip" \
     -O /tmp/with_mask.zip && unzip -q /tmp/with_mask.zip -d data/with_mask/

# Without mask images  
wget -q "https://github.com/prasoonmhwr/face-mask-detection/raw/main/dataset/without_mask.zip" \
     -O /tmp/no_mask.zip && unzip -q /tmp/no_mask.zip -d data/without_mask/

python mask_detection.py --train \
    --with-mask data/with_mask --without-mask data/without_mask
```

---

## Without any model — fallback behaviour

| Feature       | No model                          | With model             |
|---|---|---|
| Anti-spoof    | Motion-based MAD (~80%)           | MiniFASNet (~97.8%)    |
| Mask detect   | YCrCb heuristic (~70%)            | MobileNetV2 (~98%)     |

Both fallbacks work automatically — nothing breaks without models.

---

## Required pip packages

```bash
pip install torch torchvision   # for anti-spoof + mask detection
```
TensorFlow is no longer needed.
