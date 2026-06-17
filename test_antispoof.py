"""
test_antispoof.py
-----------------
Tests MiniFASNet anti-spoof on live camera.
Hold a printed photo or phone screen → should show SPOOF DETECTED.
Your real face → should show LIVE.

Press Q to quit.
"""
import cv2, sys, time
try:
    from anti_spoof import AntiSpoofChecker
except ModuleNotFoundError:
    from core.anti_spoof import AntiSpoofChecker

checker      = AntiSpoofChecker()
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

print(f"[INFO] Anti-spoof mode: {checker.mode}")
if checker.mode == "Motion":
    print("[WARNING] MiniFASNet models not found — falling back to motion detection.")
    print("          Motion detection CANNOT reliably detect phone/photo spoofs.")
    print("          Ensure models/2.7_80x80_MiniFASNetV2.pth exists.")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

WIN = "Anti-Spoof Test [Q to quit]"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN, 1280, 720)
print("[INFO] Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        time.sleep(0.04); continue

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

    for (x, y, w, h) in faces:
        roi            = frame[y:y+h, x:x+w]
        is_live, score = checker.check(roi)

        label = "LIVE" if is_live else "SPOOF DETECTED"
        color = (0, 220, 0) if is_live else (0, 60, 255)

        cv2.rectangle(frame, (x, y),      (x+w, y+h), color, 2)
        cv2.rectangle(frame, (x, y - 38), (x+w, y),   color, cv2.FILLED)
        cv2.putText(frame, f"{label}  ({score:.2f})",
                    (x + 4, y - 10),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 1)

    cv2.putText(frame, f"Mode: {checker.mode}",
                (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow(WIN, frame)
    if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
        break

cap.release()
cv2.destroyAllWindows()
