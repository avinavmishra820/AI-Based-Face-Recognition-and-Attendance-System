"""
metrics.py
----------
Reads ``ground_truth.csv`` (columns: Image, Actual, Predicted) and prints
classification metrics:

  • Accuracy
  • Precision (macro)
  • Recall    (macro)
  • F1 Score  (macro)
  • Per-class breakdown

``ground_truth.csv`` format::

    Image,Actual,Predicted
    frame_001.jpg,Alice,Alice
    frame_002.jpg,Bob,Unknown
    ...

Run:
    python metrics.py
"""

import os
import sys

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

# ---------------------------------------------------------------------------
EXPORTS_DIR       = "exports"
GROUND_TRUTH_FILE = os.path.join(EXPORTS_DIR, "ground_truth.csv")
ANALYTICS_DIR     = "analytics"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def compute_metrics(gt_path: str = GROUND_TRUTH_FILE) -> dict | None:
    if not os.path.exists(gt_path):
        print(f"[ERROR] '{gt_path}' not found.")
        print("        Create a CSV with columns: Image, Actual, Predicted")
        return None

    df = pd.read_csv(gt_path)
    required_cols = {"Actual", "Predicted"}
    if not required_cols.issubset(df.columns):
        print(f"[ERROR] '{gt_path}' must contain columns: {required_cols}")
        return None

    if df.empty:
        print("[INFO] Ground truth file is empty.")
        return None

    y_true = df["Actual"].astype(str).tolist()
    y_pred = df["Predicted"].astype(str).tolist()

    accuracy  = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall    = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1        = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print("\n" + "=" * 50)
    print("  RECOGNITION PERFORMANCE METRICS")
    print("=" * 50)
    print(f"  Accuracy  : {accuracy  * 100:.2f}%")
    print(f"  Precision : {precision * 100:.2f}%  (macro)")
    print(f"  Recall    : {recall    * 100:.2f}%  (macro)")
    print(f"  F1 Score  : {f1        * 100:.2f}%  (macro)")
    print("=" * 50)
    print("\nPer-class Report:\n")
    print(classification_report(y_true, y_pred, zero_division=0))

    # Save summary to text file
    ensure_dir(ANALYTICS_DIR)
    summary_path = os.path.join(ANALYTICS_DIR, "metrics_summary.txt")
    with open(summary_path, "w") as fh:
        fh.write("RECOGNITION PERFORMANCE METRICS\n")
        fh.write("=" * 50 + "\n")
        fh.write(f"Accuracy  : {accuracy  * 100:.2f}%\n")
        fh.write(f"Precision : {precision * 100:.2f}%  (macro)\n")
        fh.write(f"Recall    : {recall    * 100:.2f}%  (macro)\n")
        fh.write(f"F1 Score  : {f1        * 100:.2f}%  (macro)\n\n")
        fh.write(classification_report(y_true, y_pred, zero_division=0))
    print(f"[INFO] Summary saved -> '{summary_path}'")

    return {
        "accuracy":  accuracy,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
    }


if __name__ == "__main__":
    result = compute_metrics()
    if result is None:
        sys.exit(1)
