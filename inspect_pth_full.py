"""
inspect_pth_full.py — dumps ALL unique layer name patterns (no tensor values)
"""
import torch, os, re

for fname in sorted(os.listdir("models")):
    if not fname.endswith(".pth"):
        continue
    path = os.path.join("models", fname)
    raw  = torch.load(path, map_location="cpu", weights_only=False)
    sd   = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    keys = [k.replace("module.", "") for k in sd.keys()]

    print(f"\n{'='*60}")
    print(f"{fname}  ({len(keys)} keys after stripping 'module.')")
    print(f"{'='*60}")
    for k in keys:
        v = sd.get("module."+k, sd.get(k))
        print(f"  {k:<55} {tuple(v.shape)}")
