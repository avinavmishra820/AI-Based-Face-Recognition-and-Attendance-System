"""
confusion_matrix_plot.py
------------------------
Reads ``ground_truth.csv`` and saves a labelled confusion matrix image to
``analytics/confusion_matrix.png``.

Run:
    python confusion_matrix_plot.py
"""

import os
import sys

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ---------------------------------------------------------------------------
EXPORTS_DIR       = "exports"
GROUND_TRUTH_FILE = os.path.join(EXPORTS_DIR, "ground_truth.csv")
ANALYTICS_DIR     = "analytics"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_confusion_matrix(gt_path: str = GROUND_TRUTH_FILE) -> str | None:
    if not os.path.exists(gt_path):
        print(f"[ERROR] '{gt_path}' not found.")
        return None

    df = pd.read_csv(gt_path)
    if df.empty or "Actual" not in df.columns or "Predicted" not in df.columns:
        print("[ERROR] CSV must have 'Actual' and 'Predicted' columns and not be empty.")
        return None

    y_true = df["Actual"].astype(str).tolist()
    y_pred = df["Predicted"].astype(str).tolist()

    labels = sorted(set(y_true + y_pred))
    cm     = confusion_matrix(y_true, y_pred, labels=labels)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    fig_size = max(6, len(labels))
    fig, ax  = plt.subplots(figsize=(fig_size, fig_size))
    disp.plot(ax=ax, colorbar=True, cmap="Blues", xticks_rotation=45)
    ax.set_title("Confusion Matrix — Face Recognition")
    plt.tight_layout()

    ensure_dir(ANALYTICS_DIR)
    output_path = os.path.join(ANALYTICS_DIR, "confusion_matrix.png")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"[INFO] Confusion matrix saved -> '{output_path}'")
    return output_path


if __name__ == "__main__":
    result = plot_confusion_matrix()
    if result is None:
        sys.exit(1)
