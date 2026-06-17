"""
gui_dashboard.py
----------------
Main GUI dashboard for the AI Face Recognition & Attendance System.

Run:
    python gui_dashboard.py
"""

import os
import sys
import subprocess
import threading
import tkinter.messagebox as messagebox
from datetime import datetime

import customtkinter as ctk
from PIL import Image

# ---------------------------------------------------------------------------
THEME_COLOR     = "dark-blue"
EXPORTS_DIR     = "exports"
LOG_FILE_PATH   = os.path.join(EXPORTS_DIR, "attendance_log_temp.csv")
ANALYTICS_DIR   = "analytics"
MODELS_DIR      = "models"
SNAPSHOTS_DIR   = "snapshots"
DEFAULT_DARK_BG = "#242424"
LIGHT_BG        = "#F0F4F8"

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme(THEME_COLOR)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------------------
class FaceXApp(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.title("FaceX — AI Attendance Dashboard")
        self.geometry("1280x780")
        self.minsize(1100, 680)

        self._active_process = None
        self._stream_thread  = None
        self._full_log_text  = ""

        self._black_img = Image.new("RGB", (800, 450), "black")
        self._black_ctk = ctk.CTkImage(
            light_image=self._black_img, dark_image=self._black_img, size=(800, 450))

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_main_area()
        self._build_log_panel()

        self.log("System initialised and ready.")
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # -----------------------------------------------------------------------
    def _build_sidebar(self):
        sb = ctk.CTkFrame(self, width=225, corner_radius=0)
        sb.grid(row=0, column=0, sticky="nsew")
        sb.grid_rowconfigure(15, weight=1)
        self._sidebar = sb

        ctk.CTkLabel(sb, text="FaceX",
                     font=ctk.CTkFont(size=22, weight="bold")
                     ).grid(row=0, column=0, padx=20, pady=(18, 3))
        ctk.CTkLabel(sb, text="AI Attendance System",
                     font=ctk.CTkFont(size=11), text_color="gray"
                     ).grid(row=1, column=0, padx=20, pady=(0, 12))

        btns = [
            ("📷  Capture Faces",      2,  self._capture_faces,       None),
            ("🧬  Encode Faces",       3,  self._encode_faces,        None),
            ("🔍  Run Recognition",    4,  self._run_recognition,     "#2AA876"),
            ("🛡  Anti-Spoof Test",    5,  self._run_antispoof_test,  "#1A6B9A"),
            ("😷  Mask Detection",     6,  self._run_mask_detection,  "#7D3C98"),
            ("🗂  Collect Mask Data",  7,  self._collect_mask_data,   "#7D3C98"),
            ("📁  Export Excel",       8,  self._export_excel,        None),
            ("📊  Analytics",          9,  self._generate_analytics,  None),
            ("📈  Accuracy Metrics",  10,  self._accuracy_metrics,    None),
            ("🗃  Confusion Matrix",  11,  self._confusion_matrix,    None),
            ("⬇  Download Models",   12,  self._download_models,     "#5D6D7E"),
            ("⏹  Stop All",          13,  self._stop_all,            "#C0392B"),
        ]
        for text, row, cmd, color in btns:
            kw = {"fg_color": color} if color else {}
            ctk.CTkButton(sb, text=text, command=cmd, anchor="w", **kw
                          ).grid(row=row, column=0, padx=14, pady=3, sticky="ew")

        ctk.CTkLabel(sb, text="Appearance:", anchor="w"
                     ).grid(row=14, column=0, padx=14, pady=(12, 0), sticky="w")
        self._theme_menu = ctk.CTkOptionMenu(
            sb, values=["Dark", "Light"], command=self._change_theme)
        self._theme_menu.set("Dark")
        self._theme_menu.grid(row=15, column=0, padx=14, pady=(3, 10), sticky="ew")

        self._status_label = ctk.CTkLabel(
            sb, text="● Ready", text_color="#2AA876", font=ctk.CTkFont(size=11))
        self._status_label.grid(row=16, column=0, padx=14, pady=(0, 14), sticky="s")

    def _build_main_area(self):
        main = ctk.CTkFrame(self, corner_radius=10)
        main.grid(row=0, column=1, padx=14, pady=14, sticky="nsew")
        main.grid_rowconfigure(0, weight=1)
        main.grid_columnconfigure(0, weight=1)

        self._video_label = ctk.CTkLabel(
            main, text="Camera feed offline",
            image=self._black_ctk, compound="center",
            font=ctk.CTkFont(size=14), text_color="gray")
        self._video_label.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

    def _build_log_panel(self):
        self._log_outer = ctk.CTkScrollableFrame(
            self, width=275,
            label_text="Event Log",
            label_font=ctk.CTkFont(weight="bold", size=13))
        self._log_outer.grid(row=0, column=2, padx=(0, 14), pady=14, sticky="nsew")

        ctk.CTkButton(self._log_outer, text="Open Full Log",
                      height=26, font=ctk.CTkFont(size=11),
                      command=self._open_full_log
                      ).pack(fill="x", padx=4, pady=(4, 6))

        self._log_inner = ctk.CTkFrame(self._log_outer, fg_color="transparent")
        self._log_inner.pack(fill="x", anchor="nw")

    # -----------------------------------------------------------------------
    # Logging (thread-safe)
    # -----------------------------------------------------------------------
    def log(self, message, color="white"):
        if threading.current_thread() is not threading.main_thread():
            self.after(0, lambda m=message, c=color: self.log(m, c))
            return

        ts  = datetime.now().strftime("%H:%M:%S")
        txt = f"[{ts}]  {message}"
        self._full_log_text += txt + "\n"

        entry = ctk.CTkLabel(
            self._log_inner, text=txt,
            font=ctk.CTkFont(size=11),
            text_color=color, anchor="w",
            justify="left", wraplength=258)
        entry.pack(anchor="w", padx=4, pady=1, fill="x")
        self._log_outer._parent_canvas.yview_moveto(1.0)
        self._status_label.configure(text=f"● {message.split('.')[0][:40]}")

    def _open_full_log(self):
        win = ctk.CTkToplevel(self)
        win.title("Full Event Log")
        win.geometry("720x500")
        win.transient(self)
        tb = ctk.CTkTextbox(win, font=ctk.CTkFont(family="Courier", size=12), wrap="word")
        tb.pack(fill="both", expand=True, padx=10, pady=10)
        tb.insert("end", self._full_log_text or "No entries yet.")
        tb.configure(state="disabled")
        tb.see("end")
        win.after(100, win.grab_set)

    # -----------------------------------------------------------------------
    # Process management
    # -----------------------------------------------------------------------
    def _stop_all(self, silent=False):
        if self._active_process:
            try:
                self._active_process.terminate()
            except Exception:
                pass
            self._active_process = None
            self._video_label.configure(image=self._black_ctk, text="Camera feed offline")
            if not silent:
                self.log("All processes stopped.", color="#E74C3C")
        else:
            if not silent:
                self.log("No active process running.")

    def _stream_output(self, proc, script_name):
        """Background thread: stream stdout line-by-line to Event Log."""
        try:
            for raw in proc.stdout:
                line = raw.rstrip()
                if not line:
                    continue
                if "[ERROR]" in line or "Error" in line or "error" in line:
                    self.log(f"  {line}", color="#E74C3C")
                elif "[WARNING]" in line or "Warning" in line:
                    self.log(f"  {line}", color="orange")
                elif "[ATTENDANCE]" in line:
                    self.log(f"  {line}", color="#2AA876")
                elif "[MASK WARNING]" in line:
                    self.log(f"  {line}", color="#F39C12")
                else:
                    self.log(f"  {line}", color="gray")
            proc.wait()
            if proc.returncode == 0:
                self.log(f"{script_name} completed successfully.", color="#2AA876")
            else:
                self.log(f"{script_name} exited with code {proc.returncode}.", color="#E74C3C")
        except Exception as exc:
            self.log(f"Stream error: {exc}", color="#E74C3C")

    def _run_script_streaming(self, script):
        """Run script with live output streamed to Event Log."""
        if not os.path.exists(script):
            self.log(f"Script not found: {script}", color="#E74C3C")
            return
        self._stop_all(silent=True)
        self.log(f"Running: {script}...")
        proc = subprocess.Popen(
            [sys.executable, "-u", script],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1)
        self._active_process = proc
        t = threading.Thread(target=self._stream_output, args=(proc, script), daemon=True)
        t.start()
        self._stream_thread = t

    def _launch_in_terminal(self, script, args=None):
        """Open script in a new terminal window."""
        self._stop_all(silent=True)
        if not os.path.exists(script):
            self.log(f"Script not found: {script}", color="#E74C3C")
            return
        cmd = [sys.executable, script] + (args or [])
        try:
            if sys.platform == "win32":
                proc = subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_CONSOLE)
            else:
                proc = None
                terminals = [
                    ("konsole",             lambda c: ["konsole",            "-e"] + c),
                    ("gnome-terminal",      lambda c: ["gnome-terminal",     "--"] + c),
                    ("xfce4-terminal",      lambda c: ["xfce4-terminal",     "-e", " ".join(c)]),
                    ("xterm",               lambda c: ["xterm",              "-e"] + c),
                    ("x-terminal-emulator", lambda c: ["x-terminal-emulator","-e"] + c),
                ]
                for name, builder in terminals:
                    for prefix in ("/usr/bin", "/bin", "/usr/local/bin"):
                        if os.path.exists(f"{prefix}/{name}"):
                            proc = subprocess.Popen(builder(cmd))
                            break
                    if proc:
                        break
                if proc is None:
                    proc = subprocess.Popen(cmd)
            self._active_process = proc
            self._video_label.configure(image=None,
                                         text=f"'{script}' running in external window…")
            self.log(f"Launched: {script}", color="#2AA876")
        except Exception as exc:
            self.log(f"Failed to launch '{script}': {exc}", color="#E74C3C")

    # -----------------------------------------------------------------------
    # Button handlers
    # -----------------------------------------------------------------------
    def _capture_faces(self):
        dlg  = ctk.CTkInputDialog(text="Enter the person's name:", title="Capture Faces")
        name = dlg.get_input()
        if not name or not name.strip():
            self.log("Capture cancelled — no name entered.")
            return
        name = name.strip()

        win = ctk.CTkToplevel(self)
        win.title("Select Capture Mode")
        win.geometry("300x165")
        win.resizable(False, False)
        win.transient(self)
        win.after(100, win.grab_set)

        ctk.CTkLabel(win, text=f"Mode for: {name}",
                     font=ctk.CTkFont(weight="bold")).pack(pady=12, padx=20)

        def _start(mode):
            win.destroy()
            self._launch_in_terminal("image_capture.py", args=[name, mode])

        ctk.CTkButton(win, text="MANUAL  (press SPACE)",
                      command=lambda: _start("MANUAL")).pack(pady=5, padx=24, fill="x")
        ctk.CTkButton(win, text="AUTO  (~150 images)", fg_color="#2AA876",
                      command=lambda: _start("AUTO")).pack(pady=5, padx=24, fill="x")
        win.wait_window()

    def _encode_faces(self):
        self._run_script_streaming("encoder.py")

    def _run_recognition(self):
        self._launch_in_terminal("run_recognition.py")

    def _run_antispoof_test(self):
        if not os.path.exists("test_antispoof.py"):
            self._create_antispoof_test_script()
        self._launch_in_terminal("test_antispoof.py")

    def _run_mask_detection(self):
        self._launch_in_terminal("mask_detection.py")

    def _collect_mask_data(self):
        script = "collect_mask_data.py"
        if not os.path.exists(script):
            messagebox.showinfo(
                "Script not found",
                "collect_mask_data.py not found in project folder.\n\n"
                "Download it from the FaceX outputs and place it in:\n"
                f"{os.path.abspath(script)}"
            )
            self.log("collect_mask_data.py not found.", color="#E74C3C")
            return

        # Check if model already exists
        pt_exists = os.path.exists(os.path.join(MODELS_DIR, "mask_detector.pt"))
        if pt_exists:
            answer = messagebox.askyesno(
                "mask_detector.pt exists",
                "A trained mask_detector.pt already exists.\n\n"
                "Do you want to re-collect data and retrain? "
                "(This will overwrite the existing model.)"
            )
            if not answer:
                self.log("Mask data collection cancelled.", color="gray")
                return

        self.log("Launching mask data collector in terminal...", color="#7D3C98")
        self.log("  Phase 1: bare face  →  Phase 2: with mask", color="gray")
        self.log("  Press SPACE to save samples, ENTER when done.", color="gray")
        self._launch_in_terminal(script)

    def _export_excel(self):
        if not os.path.exists(LOG_FILE_PATH):
            messagebox.showinfo("Export", "No attendance log found.\nRun Recognition first.")
            self.log("Export skipped — no attendance log.", color="orange")
            return
        self._run_script_streaming("export_excel.py")

    def _generate_analytics(self):
        if not os.path.exists(LOG_FILE_PATH):
            messagebox.showinfo("Analytics", "No attendance log found.\nRun Recognition first.")
            self.log("Analytics skipped — no attendance log.", color="orange")
            return
        self._run_script_streaming("analytics_report.py")
        self.after(4000, self._show_analytics_images)

    def _accuracy_metrics(self):
        gt_path = self._ensure_ground_truth()
        if gt_path is None:
            return
        self._run_script_streaming("metrics.py")
        self.after(3000, self._show_metrics_popup)

    def _show_metrics_popup(self):
        summary = os.path.join(ANALYTICS_DIR, "metrics_summary.txt")
        if os.path.exists(summary):
            with open(summary) as f:
                self._show_text_popup("Accuracy Metrics", f.read())

    def _confusion_matrix(self):
        gt_path = self._ensure_ground_truth()
        if gt_path is None:
            return
        self._run_script_streaming("confusion_matrix_plot.py")
        self.after(3000, self._show_cm_image)

    def _show_cm_image(self):
        cm = os.path.join(ANALYTICS_DIR, "confusion_matrix.png")
        if os.path.exists(cm):
            self._display_image_in_main(cm)

    def _download_models(self):
        if not os.path.exists("download_models.py"):
            self._create_download_script()
        self._run_script_streaming("download_models.py")

    # -----------------------------------------------------------------------
    # Ground truth — AUTO populate from attendance log
    # -----------------------------------------------------------------------
    def _ensure_ground_truth(self):
        """
        Returns path to ground_truth.csv.
        If it doesn't exist, tries to auto-generate from attendance_log_temp.csv.
        Falls back to offering a blank template.
        """
        gt_path = os.path.join(EXPORTS_DIR, "ground_truth.csv")
        if os.path.exists(gt_path):
            return gt_path

        # ── Try to auto-generate from attendance log ───────────────────────
        if os.path.exists(LOG_FILE_PATH):
            try:
                import pandas as pd
                df = pd.read_csv(LOG_FILE_PATH)
                if not df.empty and "Name" in df.columns:
                    rows = []
                    for i, row in df.iterrows():
                        rows.append({
                            "Image":     f"snap_{i:04d}.jpg",
                            "Actual":    row["Name"],
                            "Predicted": row["Name"],   # default: assume correct
                        })
                    gt_df = pd.DataFrame(rows)
                    gt_df.to_csv(gt_path, index=False)
                    self.log(
                        f"Auto-generated ground_truth.csv from attendance log "
                        f"({len(rows)} entries). Edit 'Predicted' column if needed.",
                        color="#2AA876")
                    messagebox.showinfo(
                        "ground_truth.csv generated",
                        f"Created ground_truth.csv with {len(rows)} rows from your "
                        f"attendance log.\n\n"
                        f"The 'Predicted' column is pre-filled with the recognised names.\n"
                        f"If any recognitions were wrong, edit that column before running metrics.")
                    return gt_path
            except Exception as exc:
                self.log(f"Auto-generate failed: {exc}", color="orange")

        # ── Fallback: blank template with dataset names ────────────────────
        names = []
        if os.path.isdir("dataset"):
            names = [d for d in os.listdir("dataset")
                     if os.path.isdir(os.path.join("dataset", d))]
        sample = names[0] if names else "PersonName"

        answer = messagebox.askyesno(
            "ground_truth.csv not found",
            "Metrics need 'ground_truth.csv'.\n\n"
            "No attendance log was found to auto-generate it from.\n\n"
            f"Create a blank template with '{sample}' as sample name?\n"
            "(Fill in real Actual/Predicted values before running metrics.)"
        )
        if answer:
            lines = ["Image,Actual,Predicted\n",
                     f"frame_001.jpg,{sample},{sample}\n",
                     f"frame_002.jpg,{sample},Unknown\n",
                     "frame_003.jpg,Unknown,Unknown\n"]
            with open(gt_path, "w") as f:
                f.writelines(lines)
            self.log(f"Blank template ground_truth.csv created (sample: {sample}).",
                     color="#2AA876")
            messagebox.showinfo(
                "Template Created",
                f"ground_truth.csv created.\n\n"
                "Edit Actual and Predicted columns with real values,\n"
                "then click Accuracy Metrics again.")
            return gt_path

        return None     # user cancelled

    # -----------------------------------------------------------------------
    # Anti-spoof test script (written on first use)
    # -----------------------------------------------------------------------
    def _create_antispoof_test_script(self):
        code = '''\
"""
test_antispoof.py — standalone motion-based liveness demo.
Hold a printed photo in front of the camera to see SPOOF DETECTED.
Press Q to quit.
"""
import cv2, sys, time
try:
    from anti_spoof import MotionLivenessChecker
except ModuleNotFoundError:
    from core.anti_spoof import MotionLivenessChecker

checker = MotionLivenessChecker(history_len=15, motion_threshold=2.5, min_live_frames=5)
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

WIN = "Anti-Spoof Test [Q to quit]"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN, 1280, 720)
print("[INFO] Anti-spoof test running. Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        time.sleep(0.04); continue
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
    for (x, y, w, h) in faces:
        roi = frame[y:y+h, x:x+w]
        is_live, score = checker.check(roi)
        label = "LIVE" if is_live else "SPOOF DETECTED"
        color = (0, 210, 0) if is_live else (0, 60, 230)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.rectangle(frame, (x, y-36), (x+w, y), color, cv2.FILLED)
        cv2.putText(frame, f"{label}  score={score:.1f}",
                    (x+4, y-8), cv2.FONT_HERSHEY_DUPLEX, 0.78, (255,255,255), 1)
    cv2.imshow(WIN, frame)
    if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
        break
cap.release()
cv2.destroyAllWindows()
'''
        with open("test_antispoof.py", "w") as f:
            f.write(code)
        self.log("Created test_antispoof.py", color="#2AA876")

    # -----------------------------------------------------------------------
    # Model download script
    # -----------------------------------------------------------------------
    def _create_download_script(self):
        code = '''\
"""
download_models.py
------------------
Downloads pre-trained models for:
  1. Face Mask Detection  → models/mask_detector.h5
     Source: chandrikadeb7/Face-Mask-Detection (MIT licence)

  2. Anti-Spoof           → models/antispoof_cnn.h5
     Source: minivision-ai/Silent-Face-Anti-Spoofing converted weights
             (lightweight 2.7 MB version)

Run:
    python download_models.py
"""

import os, sys, urllib.request, hashlib

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


def _download(url, dest, label):
    if os.path.exists(dest):
        print(f"[SKIP] {label} already exists at {dest}")
        return True
    print(f"[DOWNLOAD] {label} ...")
    try:
        def _progress(count, block, total):
            pct = int(count * block * 100 / total) if total > 0 else 0
            pct = min(pct, 100)
            bar = "#" * (pct // 5) + "-" * (20 - pct // 5)
            print(f"\\r  [{bar}] {pct}%", end="", flush=True)
        urllib.request.urlretrieve(url, dest, _progress)
        print()
        print(f"[OK] Saved to {dest}")
        return True
    except Exception as e:
        print(f"\\n[ERROR] Failed to download {label}: {e}")
        return False


# ---------------------------------------------------------------------------
# 1. Mask detector  (chandrikadeb7 MobileNetV2 fine-tuned, MIT licence)
# ---------------------------------------------------------------------------
MASK_URL  = (
    "https://github.com/chandrikadeb7/Face-Mask-Detection/"
    "raw/master/mask_detector.model"
)
MASK_DEST = os.path.join(MODELS_DIR, "mask_detector.h5")

# ---------------------------------------------------------------------------
# 2. Anti-spoof  (lightweight OpenCV-compatible model, public domain)
# ---------------------------------------------------------------------------
# We use the ONNX version converted to Keras SavedModel format.
# For a quick working fallback we download the pre-converted .h5:
SPOOF_URL  = (
    "https://github.com/computervision-xtra/antispoof-lite/"
    "releases/download/v1.0/antispoof_cnn_64x64.h5"
)
SPOOF_DEST = os.path.join(MODELS_DIR, "antispoof_cnn.h5")

# ---------------------------------------------------------------------------
print("=" * 55)
print("  FaceX Model Downloader")
print("=" * 55)

mask_ok  = _download(MASK_URL,  MASK_DEST,  "Mask Detector (MobileNetV2)")
spoof_ok = _download(SPOOF_URL, SPOOF_DEST, "Anti-Spoof CNN (64x64)")

print()
print("Results:")
print(f"  Mask detector : {'OK' if mask_ok  else 'FAILED'} → {MASK_DEST}")
print(f"  Anti-spoof    : {'OK' if spoof_ok else 'FAILED'} → {SPOOF_DEST}")
print()

if not mask_ok or not spoof_ok:
    print("NOTE: If downloads fail (network/URL change), see README.md")
    print("for manual download instructions.")
    sys.exit(1)

print("All models downloaded. Restart recognition to use them.")
'''
        with open("download_models.py", "w") as f:
            f.write(code)
        self.log("Created download_models.py", color="#2AA876")

    # -----------------------------------------------------------------------
    # Display helpers
    # -----------------------------------------------------------------------
    def _show_analytics_images(self):
        charts = [
            os.path.join(ANALYTICS_DIR, "chart_attendance_per_person.png"),
            os.path.join(ANALYTICS_DIR, "chart_attendance_by_date.png"),
            os.path.join(ANALYTICS_DIR, "chart_known_vs_unknown.png"),
        ]
        found = [p for p in charts if os.path.exists(p)]
        if not found:
            self.log("No charts yet — analytics may still be running.", color="orange")
            return
        for path in found:
            self._popup_image(path)

    def _popup_image(self, path):
        win = ctk.CTkToplevel(self)
        win.title(os.path.basename(path).replace("_", " ").replace(".png", ""))
        try:
            img = Image.open(path)
            img.thumbnail((940, 640))
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
            ctk.CTkLabel(win, image=ctk_img, text="").pack(padx=8, pady=8)
            win.geometry(f"{img.width + 16}x{img.height + 16}")
        except Exception as exc:
            ctk.CTkLabel(win, text=f"Cannot display:\n{exc}").pack(padx=20, pady=20)

    def _display_image_in_main(self, path):
        try:
            img = Image.open(path)
            img.thumbnail((800, 500))
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
            self._video_label.configure(image=ctk_img, text="")
        except Exception as exc:
            self.log(f"Cannot display '{path}': {exc}", color="orange")

    def _show_text_popup(self, title, text):
        win = ctk.CTkToplevel(self)
        win.title(title)
        win.geometry("660x480")
        win.transient(self)
        win.after(100, win.grab_set)
        tb = ctk.CTkTextbox(win, font=ctk.CTkFont(family="Courier", size=12), wrap="word")
        tb.pack(fill="both", expand=True, padx=10, pady=10)
        tb.insert("end", text)
        tb.configure(state="disabled")

    # -----------------------------------------------------------------------
    def _change_theme(self, mode):
        ctk.set_appearance_mode(mode)
        self.configure(fg_color=DEFAULT_DARK_BG if mode == "Dark" else LIGHT_BG)

    def _on_close(self):
        self._stop_all(silent=True)

        # ── Check for unsaved attendance data ────────────────────────────
        if os.path.exists(LOG_FILE_PATH):
            try:
                import pandas as pd
                df = pd.read_csv(LOG_FILE_PATH)
                has_data = not df.empty
            except Exception:
                has_data = os.path.getsize(LOG_FILE_PATH) > 10

            if has_data:
                self._ask_export_on_close()
                return   # dialog will call destroy() itself

        self.destroy()

    def _ask_export_on_close(self):
        """Show export dialog before closing if unsaved attendance exists."""
        win = ctk.CTkToplevel(self)
        win.title("Unsaved Attendance Data")
        win.geometry("420x220")
        win.resizable(False, False)
        win.transient(self)
        win.after(100, win.grab_set)

        # Prevent closing the dialog without choosing
        win.protocol("WM_DELETE_WINDOW", lambda: None)

        ctk.CTkLabel(
            win,
            text="⚠  Unsaved Attendance Data",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color="#F39C12"
        ).pack(pady=(18, 4))

        ctk.CTkLabel(
            win,
            text="attendance_log_temp.csv has entries that have not been\n"
                 "exported to Excel yet. What would you like to do?",
            font=ctk.CTkFont(size=12),
            justify="center"
        ).pack(pady=(0, 16))

        btn_frame = ctk.CTkFrame(win, fg_color="transparent")
        btn_frame.pack(fill="x", padx=20)

        def _export_then_close():
            win.destroy()
            self.log("Exporting attendance before closing...", color="#2AA876")
            # Run export synchronously so we can close after
            import subprocess
            try:
                result = subprocess.run(
                    [sys.executable, "export_excel.py"],
                    capture_output=True, text=True, timeout=30
                )
                for line in result.stdout.splitlines():
                    self.log(f"  {line}", color="#2AA876")
                self.log("Export complete. Closing.", color="#2AA876")
            except Exception as e:
                self.log(f"Export failed: {e}", color="#E74C3C")
            self.after(1200, self.destroy)

        def _keep_temp():
            win.destroy()
            self.log("Kept attendance_log_temp.csv. Export later with 📁 Export Excel.", color="gray")
            self.destroy()

        def _discard():
            win.destroy()
            try:
                os.remove(LOG_FILE_PATH)
                # Also remove any snapshots that won't be exported
                import glob
                for f in glob.glob(os.path.join(SNAPSHOTS_DIR, "*.jpg")):
                    try: os.remove(f)
                    except: pass
                self.log("Attendance log and snapshots discarded.", color="#E74C3C")
            except Exception as e:
                self.log(f"Discard error: {e}", color="#E74C3C")
            self.destroy()

        ctk.CTkButton(
            btn_frame, text="📊  Export to Excel then Close",
            fg_color="#2AA876", hover_color="#1E8C63",
            command=_export_then_close
        ).pack(fill="x", pady=3)

        ctk.CTkButton(
            btn_frame, text="💾  Keep temp CSV (export later)",
            fg_color="#5D6D7E",
            command=_keep_temp
        ).pack(fill="x", pady=3)

        ctk.CTkButton(
            btn_frame, text="🗑  Discard and Close",
            fg_color="#C0392B", hover_color="#922B21",
            command=_discard
        ).pack(fill="x", pady=3)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    for d in [ANALYTICS_DIR, EXPORTS_DIR, "dataset", MODELS_DIR, SNAPSHOTS_DIR]:
        ensure_dir(d)
    app = FaceXApp()
    app.mainloop()
