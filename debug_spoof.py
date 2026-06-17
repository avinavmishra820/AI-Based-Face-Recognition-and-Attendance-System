"""
debug_spoof.py — captures one frame, runs MiniFASNet, prints raw scores.
Run: python debug_spoof.py
Press SPACE to capture and score, Q to quit.
"""
import cv2, os, sys, numpy as np

MODELS_DIR = "models"

# ── Load models inline (same arch as anti_spoof.py) ──────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F

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

class DWRSE(nn.Module):
    def __init__(self, in_c, out_c, g, reduction, s=1):
        super().__init__()
        self.use_res = (s == 1 and in_c == out_c)
        self.conv    = CBP(in_c, g, 1, 1, 0)
        self.conv_dw = CBP(g, g, 3, s, 1, g=g)
        self.project = Project(g, out_c)
        self.se_fc1  = nn.Conv2d(out_c, reduction, 1, bias=False)
        self.se_bn1  = nn.BatchNorm2d(reduction)
        self.se_fc2  = nn.Conv2d(reduction, out_c, 1, bias=False)
        self.se_bn2  = nn.BatchNorm2d(out_c)
    def _se(self, x):
        s = x.mean(dim=(2,3), keepdim=True)
        s = torch.relu(self.se_bn1(self.se_fc1(s)))
        s = torch.sigmoid(self.se_bn2(self.se_fc2(s)))
        return x * s
    def forward(self, x):
        out = self.project(self.conv_dw(self.conv(x)))
        out = self._se(out)
        return out + x if self.use_res else out

class ResStack(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.model = nn.ModuleList(blocks)
    def forward(self, x):
        for b in self.model: x = b(x)
        return x

class MiniFASNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=CBP(3,32,3,1,1); self.conv2_dw=CBP(32,32,3,1,1,g=32)
        self.conv_23=DW(32,64,g=103,s=2)
        self.conv_3=ResStack([DWR(64,64,g=13)]*4)
        self.conv_34=DW(64,128,g=231,s=2)
        self.conv_4=ResStack([DWR(128,128,g=231),DWR(128,128,g=52),DWR(128,128,g=26),DWR(128,128,g=77),DWR(128,128,g=26),DWR(128,128,g=26)])
        self.conv_45=DW(128,128,g=308,s=1)
        self.conv_5=ResStack([DWR(128,128,g=26)]*2)
        self.conv_6_sep=CBP(128,512,1,1,0); self.conv_6_dw=DWHead(512,k=5)
        self.linear=nn.Linear(512,128,bias=False)  # bias=False — ckpt has no linear.bias
        self.bn=nn.BatchNorm1d(128); self.prob=nn.Linear(128,3,bias=False)
    def forward(self,x):
        x=self.conv1(x);x=self.conv2_dw(x);x=self.conv_23(x);x=self.conv_3(x)
        x=self.conv_34(x);x=self.conv_4(x);x=self.conv_45(x);x=self.conv_5(x)
        x=self.conv_6_sep(x);x=self.conv_6_dw(x)
        x=F.adaptive_avg_pool2d(x,1).flatten(1)
        return self.prob(self.bn(self.linear(x)))

class MiniFASNetV1SE(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=CBP(3,32,3,1,1); self.conv2_dw=CBP(32,32,3,1,1,g=32)
        self.conv_23=DW(32,64,g=103,s=2)
        self.conv_3=ResStack([DWR(64,64,g=13),DWR(64,64,g=26),DWR(64,64,g=13),DWRSE(64,64,g=52,reduction=16)])
        self.conv_34=DW(64,128,g=231,s=2)
        self.conv_4=ResStack([DWR(128,128,g=154),DWR(128,128,g=52),DWR(128,128,g=26),DWR(128,128,g=52),DWR(128,128,g=26),DWRSE(128,128,g=26,reduction=32)])
        self.conv_45=DW(128,128,g=308,s=1)
        self.conv_5=ResStack([DWR(128,128,g=26),DWRSE(128,128,g=26,reduction=32)])
        self.conv_6_sep=CBP(128,512,1,1,0); self.conv_6_dw=DWHead(512,k=5)
        self.linear=nn.Linear(512,128,bias=False)  # bias=False — ckpt has no linear.bias
        self.bn=nn.BatchNorm1d(128); self.prob=nn.Linear(128,3,bias=False)
    def forward(self,x):
        x=self.conv1(x);x=self.conv2_dw(x);x=self.conv_23(x);x=self.conv_3(x)
        x=self.conv_34(x);x=self.conv_4(x);x=self.conv_45(x);x=self.conv_5(x)
        x=self.conv_6_sep(x);x=self.conv_6_dw(x)
        x=F.adaptive_avg_pool2d(x,1).flatten(1)
        return self.prob(self.bn(self.linear(x)))

# ── Load ──────────────────────────────────────────────────────────────────
models = []
for fname, cls in [("2.7_80x80_MiniFASNetV2.pth", MiniFASNetV2),
                   ("4_0_0_80x80_MiniFASNetV1SE.pth", MiniFASNetV1SE)]:
    path = os.path.join(MODELS_DIR, fname)
    m = cls()
    raw = torch.load(path, map_location="cpu", weights_only=False)
    sd  = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    sd  = {k.replace("module.",""):v for k,v in sd.items()}
    miss, unexp = m.load_state_dict(sd, strict=False)
    print(f"{fname}: missing={miss}, unexpected={len(unexp)}")
    m.eval(); models.append((fname, m))

# ── Run on webcam ─────────────────────────────────────────────────────────
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)
print("\nSPACE = score current frame | Q = quit")

MEAN = np.array([0.485,0.456,0.406], dtype=np.float32)
STD  = np.array([0.229,0.224,0.225], dtype=np.float32)

while True:
    ret, frame = cap.read()
    if not ret: break
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60,60))
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
    cv2.putText(frame, "SPACE=score  Q=quit", (10,25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
    cv2.imshow("debug_spoof", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'): break
    if key == ord(' '):
        if len(faces) == 0:
            print("No face detected — move closer"); continue
        x,y,w,h = faces[0]
        roi = frame[y:y+h, x:x+w]
        img = cv2.resize(roi, (80,80))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype("float32")/255.0
        rgb = (rgb - MEAN) / STD
        t   = torch.from_numpy(rgb.transpose(2,0,1)).unsqueeze(0).float()

        print("\n--- Scores ---")
        all_real = []
        with torch.no_grad():
            for fname, m in models:
                logits = m(t)
                probs  = F.softmax(logits, dim=1)[0]
                real   = probs[2].item()
                all_real.append(real)
                print(f"  {fname}")
                print(f"    logits : {logits[0].tolist()}")
                print(f"    softmax: spoof1={probs[0]:.4f}  spoof2={probs[1]:.4f}  REAL={probs[2]:.4f}{probs[2]:.4f}")
        avg = float(np.mean(all_real))
        verdict = "REAL ✓" if avg >= 0.60 else f"SPOOF ✗  (avg REAL[2]={avg:.4f}, need ≥0.60)"
        print(f"  VERDICT: {verdict}")
        print(f"  → If real score is ~0.33, model is outputting uniform — weights not loaded right")
        print(f"  → If real score is low but not 0.33, try lowering threshold in anti_spoof.py")

cap.release(); cv2.destroyAllWindows()
