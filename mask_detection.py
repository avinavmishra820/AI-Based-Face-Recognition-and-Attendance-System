"""
mask_detection.py — PyTorch only, no TensorFlow.
Press Q to quit.  Train: python mask_detection.py --train --with-mask X --without-mask Y
"""
import os, sys, argparse, time, cv2, numpy as np
MODEL_PATH    = os.path.join("models", "mask_detector.pt")
INPUT_SIZE   = (224,224)
THRESHOLD    = 0.55
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")

def _build_model():
    import torch.nn as nn
    from torchvision import models
    m = models.mobilenet_v2(weights=None)
    m.classifier[1] = nn.Linear(m.last_channel, 2)
    return m

def load_model(path=MODEL_PATH):
    """Load PyTorch mask_detector.pt. Train with: python collect_mask_data.py"""
    if not os.path.exists(path):
        print(f"[MASK] No model at '{path}' — using skin-tone heuristic.")
        print("[MASK] Train one with: python collect_mask_data.py")
        return None
    try:
        import torch
        model = _build_model()
        state = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state); model.eval()
        print("[MASK] PyTorch mask model loaded.")
        return model
    except Exception as e:
        print(f"[MASK] Load failed: {e}")
        return None
def _pre(roi):
    import torch
    img = cv2.resize(roi,INPUT_SIZE)
    rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB).astype("float32")/255.0
    rgb = (rgb-[0.485,0.456,0.406])/[0.229,0.224,0.225]
    return torch.from_numpy(rgb.transpose(2,0,1)).unsqueeze(0).float()

def predict_cnn(model,roi):
    import torch,torch.nn.functional as F
    with torch.no_grad():
        p=F.softmax(model(_pre(roi)),dim=1)[0]
    v=float(p[1]); return v>=THRESHOLD,v

def predict_heuristic(roi):
    if roi is None or roi.size==0: return False,0.5
    h,w=roi.shape[:2]; lower=roi[h//2:,:]
    y=cv2.cvtColor(lower,cv2.COLOR_BGR2YCrCb)
    sk=cv2.inRange(y,np.array([0,133,77],np.uint8),np.array([255,173,127],np.uint8))
    r=np.count_nonzero(sk)/(lower.shape[0]*lower.shape[1]+1e-6)
    w2=r<0.25; return w2,float(1-r if w2 else r)

def train(with_dir,no_dir,save=MODEL_PATH,epochs=15):
    import torch,torch.nn as nn,random
    from torch.utils.data import DataLoader,Dataset
    from torchvision import models,transforms
    from PIL import Image as PI
    os.makedirs(os.path.dirname(save) or ".",exist_ok=True)
    aug=transforms.Compose([transforms.Resize((256,256)),transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),transforms.ColorJitter(.2,.2,.2),
        transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    vtf=transforms.Compose([transforms.Resize(INPUT_SIZE),transforms.ToTensor(),
        transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    class DS(Dataset):
        def __init__(self,items,tf): self.items=items; self.tf=tf
        def __len__(self): return len(self.items)
        def __getitem__(self,i):
            p,l=self.items[i]; return self.tf(PI.open(p).convert("RGB")),l
    items=[(os.path.join(d,f),lb) for d,lb in[(no_dir,0),(with_dir,1)]
           for f in os.listdir(d) if os.path.isfile(os.path.join(d,f))]
    random.shuffle(items); sp=int(.8*len(items))
    tr=DataLoader(DS(items[:sp],aug),32,shuffle=True,num_workers=2)
    vl=DataLoader(DS(items[sp:],vtf),32,num_workers=2)
    dev="cuda" if torch.cuda.is_available() else "cpu"
    print(f"[MASK] Training on {dev} | {sp} train, {len(items)-sp} val")
    m=models.mobilenet_v2(weights="IMAGENET1K_V1")
    m.classifier[1]=nn.Linear(m.last_channel,2); m=m.to(dev)
    opt=torch.optim.Adam(m.parameters(),lr=1e-4)
    sch=torch.optim.lr_scheduler.StepLR(opt,5,.5)
    ce=nn.CrossEntropyLoss(); best=0.0
    for ep in range(1,epochs+1):
        m.train(); tl=0
        for X,y in tr:
            X,y=X.to(dev),y.to(dev); opt.zero_grad()
            loss=ce(m(X),y); loss.backward(); opt.step(); tl+=loss.item()
        m.eval(); c=t2=0
        with torch.no_grad():
            for X,y in vl:
                X,y=X.to(dev),y.to(dev); c+=(m(X).argmax(1)==y).sum().item(); t2+=y.size(0)
        acc=c/t2 if t2 else 0
        print(f"  Epoch {ep:02d}/{epochs} loss={tl/len(tr):.4f} val={acc*100:.1f}%")
        if acc>best: best=acc; torch.save(m.state_dict(),save); print(f"  -> Best saved")
        sch.step()
    print(f"[MASK] Done. Best {best*100:.1f}% -> {save}")

def open_camera():
    """Cross-platform robust camera initialization"""
    
    if os.name == "nt":
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    else:
        backends = [cv2.CAP_ANY]

    for backend in backends:
        for i in range(3):
            cap = cv2.VideoCapture(i, backend)

            if not cap.isOpened():
                continue

            time.sleep(0.7)  # better warm-up

            ret, frame = cap.read()

            if ret and frame is not None and frame.size > 0:
                print(f"[INFO] Camera opened (index={i}, backend={backend})")
                return cap

            cap.release()

    return None


def run():
    model = load_model()
    use_cnn = model is not None

    cap = open_camera()
    if cap is None:
        print("[ERROR] No working camera found.")
        return

    # Resolution setup (safe)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # ⚠️ FOURCC can break on some Windows setups → optional
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    except:
        pass

    WIN = "Mask Detection [Q to quit]"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 720)

    mode = "CNN" if use_cnn else "Heuristic"
    print(f"[MASK] Running ({mode}). Q to quit.")

    fc = 0
    fs = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()

        # 🔴 CRITICAL FIX: prevent OpenCV crash
        if not ret or frame is None or frame.size == 0:
            print("[WARN] Empty frame received")
            time.sleep(0.03)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

        for (x, y, w, h) in faces:
            roi = frame[y:y+h, x:x+w]
            if roi.size == 0:
                continue

            wearing, conf = (
                predict_cnn(model, roi)
                if use_cnn else
                predict_heuristic(roi)
            )

            label = "With Mask" if wearing else "No Mask"
            color = (0, 200, 0) if wearing else (0, 0, 220)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.rectangle(frame, (x, y-36), (x+w, y), color, cv2.FILLED)

            cv2.putText(
                frame,
                f"{label} {conf*100:.0f}%",
                (x+4, y-8),
                cv2.FONT_HERSHEY_DUPLEX,
                0.78,
                (255, 255, 255),
                1
            )

        fc += 1
        if time.time() - fs >= 1.0:
            fps = fc / (time.time() - fs)
            fc = 0
            fs = time.time()

        cv2.putText(
            frame,
            f"FPS:{fps:.0f} [{mode}]",
            (frame.shape[1] - 260, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        cv2.imshow(WIN, frame)

        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--train",action="store_true")
    p.add_argument("--with-mask",default="")
    p.add_argument("--without-mask",default="")
    p.add_argument("--epochs",type=int,default=15)
    a=p.parse_args()
    if a.train:
        if not a.with_mask or not a.without_mask: print("[ERROR] Need --with-mask and --without-mask"); sys.exit(1)
        train(a.with_mask,a.without_mask,epochs=a.epochs)
    else: run()
