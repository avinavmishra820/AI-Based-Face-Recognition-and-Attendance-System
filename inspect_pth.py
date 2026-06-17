"""
inspect_pth.py — run this once to see the exact layer names in the .pth files
"""
import torch, os

for fname in os.listdir("models"):
    if not fname.endswith(".pth"):
        continue
    path = os.path.join("models", fname)
    print(f"\n{'='*60}")
    print(f"FILE: {fname}")
    print(f"{'='*60}")
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict):
        sd = raw.get("state_dict", raw)
        keys = list(sd.keys())
        print(f"Total keys: {len(keys)}")
        print("First 20 keys:")
        for k in keys[:20]:
            print(f"  {k}  →  {sd[k].shape}")
        print("Last 5 keys:")
        for k in keys[-5:]:
            print(f"  {k}  →  {sd[k].shape}")
    else:
        print(f"Type: {type(raw)}")
        if hasattr(raw, "state_dict"):
            sd = raw.state_dict()
            keys = list(sd.keys())
            print(f"Total keys: {len(keys)}")
            for k in keys[:20]:
                print(f"  {k}  →  {sd[k].shape}")
