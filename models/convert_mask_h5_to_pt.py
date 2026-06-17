"""
convert_mask_h5_to_pt.py
------------------------
Converts mask_detector.h5 (Keras) → mask_detector.pt (PyTorch).

Run:  python convert_mask_h5_to_pt.py

Requires: pip install tensorflow torch torchvision numpy
"""
import os, sys, numpy as np

H5_PATH = os.path.join("models", "mask_detector.h5")
PT_PATH = os.path.join("models", "mask_detector.pt")

# ── Step 1: Inspect the .h5 ──────────────────────────────────────────────
print("=" * 60)
print("Step 1: Inspecting", H5_PATH)
print("=" * 60)

try:
    import h5py
    with h5py.File(H5_PATH, "r") as f:
        print("Keras version :", f.attrs.get("keras_version", "unknown"))
        print("Backend       :", f.attrs.get("backend", "unknown"))
        if "model_config" in f.attrs:
            import json
            cfg    = json.loads(f.attrs["model_config"])
            layers = cfg.get("config", {}).get("layers", [])
            print(f"Architecture  : {cfg.get('class_name')} — {len(layers)} layers")
            for l in layers:
                ltype = l.get("class_name", "?")
                lname = l.get("config", {}).get("name", "?")
                units = l.get("config", {}).get("units", "")
                filt  = l.get("config", {}).get("filters", "")
                info  = f"units={units}" if units else (f"filters={filt}" if filt else "")
                print(f"  {ltype:<30} {lname:<30} {info}")
except Exception as e:
    print(f"h5py inspect failed: {e}")

# ── Step 2: Try loading with TensorFlow/Keras ────────────────────────────
print("\n" + "=" * 60)
print("Step 2: Loading with TensorFlow")
print("=" * 60)

keras_model = None
try:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf
    print("TensorFlow version:", tf.__version__)
    keras_model = tf.keras.models.load_model(H5_PATH)
    print("Model loaded successfully!")
    keras_model.summary()
except Exception as e:
    print(f"TF load failed: {e}")
    print("\nTrying legacy keras load...")
    try:
        import keras
        keras_model = keras.models.load_model(H5_PATH)
        print("Loaded with standalone keras!")
        keras_model.summary()
    except Exception as e2:
        print(f"keras load also failed: {e2}")

if keras_model is None:
    print("\n[ERROR] Could not load .h5 — TF/Keras version mismatch.")
    print("Options:")
    print("  1. Run: pip install tensorflow==2.12  (if you have TF2)")
    print("  2. Run: pip install tf-keras           (Keras 2 compatibility shim)")
    print("  3. Use collect_mask_data.py to train a fresh PyTorch model instead")
    sys.exit(1)

# ── Step 3: Extract weights and match to MobileNetV2 ────────────────────
print("\n" + "=" * 60)
print("Step 3: Converting weights to PyTorch")
print("=" * 60)

try:
    import torch
    import torch.nn as nn
    from torchvision import models

    # Build PyTorch MobileNetV2 with 2-class head (same as mask_detection.py)
    pt_model = models.mobilenet_v2(weights=None)
    pt_model.classifier[1] = nn.Linear(pt_model.last_channel, 2)

    # Check if the keras model is also MobileNetV2-based
    keras_layers = [(l.name, l.__class__.__name__) for l in keras_model.layers]
    print(f"Keras model has {len(keras_layers)} layers")
    
    # Check output shape to determine number of classes
    out_shape = keras_model.output_shape
    print(f"Output shape: {out_shape}")
    n_classes = out_shape[-1]
    print(f"Number of classes: {n_classes}")

    if n_classes == 2:
        print("✓ 2-class output (with_mask / without_mask) — matches our architecture")
    else:
        print(f"⚠ {n_classes}-class output — may need architecture adjustment")
        pt_model.classifier[1] = nn.Linear(pt_model.last_channel, n_classes)

    # Try direct weight transfer if architectures match
    keras_weights = keras_model.get_weights()
    print(f"\nKeras weight tensors: {len(keras_weights)}")
    
    # Count PyTorch params
    pt_params = sum(1 for _ in pt_model.parameters())
    print(f"PyTorch param tensors: {pt_params}")

    # Save the keras model info for manual inspection
    print("\nKeras layer shapes (first 10):")
    for i, w in enumerate(keras_weights[:10]):
        print(f"  [{i}] shape={w.shape}")

    print("\n[INFO] Direct weight transfer between Keras MobileNetV2 and")
    print("       PyTorch MobileNetV2 requires careful layer mapping.")
    print("       Attempting automatic conversion via ONNX...")

except Exception as e:
    print(f"Weight extraction failed: {e}")

# ── Step 4: Convert via ONNX (most reliable path) ───────────────────────
print("\n" + "=" * 60)
print("Step 4: Keras → ONNX → PyTorch")
print("=" * 60)

try:
    import tf2onnx, onnx
    import tempfile

    onnx_path = os.path.join("models", "mask_detector_temp.onnx")
    
    # Convert Keras → ONNX
    print("Converting Keras → ONNX...")
    import tf2onnx.convert
    input_sig = [tf.TensorSpec(keras_model.input_shape, tf.float32, name="input")]
    onnx_model, _ = tf2onnx.convert.from_keras(keras_model, input_signature=input_sig)
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"ONNX saved → {onnx_path}")

    # Load ONNX as PyTorch via onnx2torch
    try:
        import onnx2torch
        onnx_loaded = onnx.load(onnx_path)
        pt_from_onnx = onnx2torch.convert(onnx_loaded)
        pt_from_onnx.eval()
        torch.save(pt_from_onnx.state_dict(), PT_PATH)
        print(f"✓ Converted! Saved → {PT_PATH}")
        os.remove(onnx_path)
    except ImportError:
        print("onnx2torch not installed.")
        print("Run: pip install onnx2torch")
        print(f"ONNX file kept at: {onnx_path}")
        print("You can load it directly with onnxruntime instead.")
        _offer_onnxruntime_fallback(onnx_path)

except ImportError as e:
    missing = str(e).split("'")[1] if "'" in str(e) else str(e)
    print(f"Missing: {missing}")
    print(f"Run: pip install tf2onnx onnx onnx2torch")
    print("\nAlternatively — fastest solution:")
    print("  python collect_mask_data.py")
    print("  (captures ~120 images each, trains in ~3 min on CPU)")
    sys.exit(0)

except Exception as e:
    print(f"ONNX conversion failed: {e}")
    print("\nFallback: train a fresh model with collect_mask_data.py")
    sys.exit(1)

print("\n" + "=" * 60)
print("DONE — mask_detector.pt ready.")
print("Run: python run_recognition.py")
print("You should see: [MASK] PyTorch model loaded.")
print("=" * 60)
