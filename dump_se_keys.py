"""Run: python dump_se_keys.py"""
import torch, os

path = os.path.join("models", "4_0_0_80x80_MiniFASNetV1SE.pth")
raw  = torch.load(path, map_location="cpu", weights_only=False)
sd   = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
sd   = {k.replace("module.", ""): v for k, v in sd.items()}

print("=== ALL conv_3.model.3 keys ===")
for k, v in sorted(sd.items()):
    if "conv_3.model.3" in k:
        print(f"  {k:<65} {tuple(v.shape)}")

print("\n=== ALL se_ keys (anywhere in model) ===")
for k, v in sorted(sd.items()):
    if "se_" in k or ".se." in k:
        print(f"  {k:<65} {tuple(v.shape)}")

print("\n=== ALL conv_4 keys (model.0 only) ===")
for k, v in sorted(sd.items()):
    if "conv_4.model.0" in k or "conv_4.model.3" in k:
        print(f"  {k:<65} {tuple(v.shape)}")
