"""
diagnose_antispoof.py
---------------------
Run this ONCE. It will:
1. Load the .pth files
2. Show which keys are missing vs what the architecture produces
3. Print the EXACT fix needed

Run:
    python diagnose_antispoof.py
"""
import os, sys
import torch
import torch.nn as nn

MODELS_DIR = "models"

# ── Same architecture as anti_spoof.py ──────────────────────────────────────
class CBP(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1, g=1):
        super().__init__()
        self.conv  = nn.Conv2d(in_c, out_c, k, s, p, groups=g, bias=False)
        self.bn    = nn.BatchNorm2d(out_c)
        self.prelu = nn.PReLU(out_c)
    def forward(self, x): return self.prelu(self.bn(self.conv(x)))

class Project(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False)
        self.bn   = nn.BatchNorm2d(out_c)
    def forward(self, x): return self.bn(self.conv(x))

class DWHead(nn.Module):
    def __init__(self, c, k=5):
        super().__init__()
        self.conv = nn.Conv2d(c, c, k, 1, k//2, groups=c, bias=False)
        self.bn   = nn.BatchNorm2d(c)
    def forward(self, x): return self.bn(self.conv(x))

class DW(nn.Module):
    def __init__(self, in_c, out_c, g, s=1):
        super().__init__()
        self.conv    = CBP(in_c, g, 1, 1, 0)
        self.conv_dw = CBP(g, g, 3, s, 1, g=g)
        self.project = Project(g, out_c)
    def forward(self, x): return self.project(self.conv_dw(self.conv(x)))

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

class ResStack(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.model = nn.ModuleList(blocks)
    def forward(self, x):
        for b in self.model: x = b(x)
        return x

class MiniFASNetV2(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.conv1      = CBP(3,   32, 3, 1, 1)
        self.conv2_dw   = CBP(32,  32, 3, 1, 1, g=32)
        self.conv_23    = DW (32,  64, g=103, s=2)
        self.conv_3     = ResStack([DWR(64,64,g=13)]*4)
        self.conv_34    = DW (64,  128, g=231, s=2)
        self.conv_4     = ResStack([DWR(128,128,g=231),DWR(128,128,g=52),
                                    DWR(128,128,g=26),DWR(128,128,g=77),
                                    DWR(128,128,g=26),DWR(128,128,g=26)])
        self.conv_45    = DW (128, 128, g=308, s=1)
        self.conv_5     = ResStack([DWR(128,128,g=26)]*2)
        self.conv_6_sep = CBP(128, 512, 1, 1, 0)
        self.conv_6_dw  = DWHead(512, k=5)
        self.linear     = nn.Linear(512, 128)
        self.bn         = nn.BatchNorm1d(128)
        self.prob       = nn.Linear(128, num_classes, bias=False)
    def forward(self, x): pass

class MiniFASNetV1SE(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.conv1      = CBP(3,   32, 3, 1, 1)
        self.conv2_dw   = CBP(32,  32, 3, 1, 1, g=32)
        self.conv_23    = DW (32,  64, g=103, s=2)
        self.conv_3     = ResStack([DWR(64,64,g=13),DWR(64,64,g=26),
                                    DWR(64,64,g=13),DWR(64,64,g=52)])
        self.conv_34    = DW (64,  128, g=231, s=2)
        self.conv_4     = ResStack([DWR(128,128,g=231),DWR(128,128,g=52),
                                    DWR(128,128,g=26),DWR(128,128,g=77),
                                    DWR(128,128,g=26),DWR(128,128,g=26)])
        self.conv_45    = DW (128, 128, g=308, s=1)
        self.conv_5     = ResStack([DWR(128,128,g=26)]*2)
        self.conv_6_sep = CBP(128, 512, 1, 1, 0)
        self.conv_6_dw  = DWHead(512, k=5)
        self.linear     = nn.Linear(512, 128)
        self.bn         = nn.BatchNorm1d(128)
        self.prob       = nn.Linear(128, num_classes, bias=False)
    def forward(self, x): pass

# ── Test loading ─────────────────────────────────────────────────────────────
pth_map = {
    "2.7_80x80_MiniFASNetV2.pth":      MiniFASNetV2,
    "4_0_0_80x80_MiniFASNetV1SE.pth":  MiniFASNetV1SE,
}

print("=" * 65)
all_ok = True
for fname, cls in pth_map.items():
    fpath = os.path.join(MODELS_DIR, fname)
    if not os.path.exists(fpath):
        print(f"[SKIP] {fname} not found in {MODELS_DIR}/")
        continue

    model = cls(num_classes=3)
    raw   = torch.load(fpath, map_location="cpu", weights_only=False)
    sd    = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    sd    = {k.replace("module.", ""): v for k, v in sd.items()}

    arch_keys = set(model.state_dict().keys())
    ckpt_keys = set(sd.keys())

    missing    = sorted(arch_keys - ckpt_keys)   # in arch but not in checkpoint
    unexpected = sorted(ckpt_keys - arch_keys)   # in checkpoint but not in arch

    print(f"\nFILE: {fname}")
    print(f"  Arch keys   : {len(arch_keys)}")
    print(f"  Ckpt keys   : {len(ckpt_keys)}")
    print(f"  Missing     : {len(missing)}")
    print(f"  Unexpected  : {len(unexpected)}")

    if missing:
        all_ok = False
        print(f"\n  MISSING (first 10) — in arch but NOT in .pth:")
        for k in missing[:10]:
            print(f"    {k}")
        print(f"\n  UNEXPECTED (first 10) — in .pth but NOT in arch:")
        for k in unexpected[:10]:
            print(f"    {k}")

        # Show side-by-side diff for first missing key
        first = missing[0]
        print(f"\n  ARCH key   : {first}")
        # Find closest match in ckpt
        prefix = ".".join(first.split(".")[:2])
        close  = [k for k in ckpt_keys if k.startswith(prefix)][:5]
        print(f"  CKPT keys with same prefix '{prefix}':")
        for k in close:
            print(f"    {k}")
    else:
        print(f"  ✓ 0 missing keys — loads perfectly!")

print("\n" + "=" * 65)
if all_ok:
    print("ALL MODELS LOAD WITH 0 MISSING KEYS — anti_spoof.py is correct.")
else:
    print("MISMATCH FOUND — paste output above to get the fix.")
