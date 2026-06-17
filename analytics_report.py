"""
analytics_report.py
-------------------
Reads ``attendance_log_temp.csv`` and generates three charts saved to
the ``analytics/`` folder:

1. Attendance count per person    — horizontal bar chart
2. Attendance over time (by date) — line chart
3. Known vs Unknown breakdown     — pie chart

Run:
    python analytics_report.py
"""

import os
import sys

import pandas as pd
import matplotlib
matplotlib.use("Agg")           # Non-interactive backend (safe for subprocess use)
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EXPORTS_DIR    = "exports"
LOG_FILE_PATH  = os.path.join(EXPORTS_DIR, "attendance_log_temp.csv")
ANALYTICS_DIR  = "analytics"
DPI            = 150


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_log() -> pd.DataFrame | None:
    if not os.path.exists(LOG_FILE_PATH):
        print(f"[ERROR] '{LOG_FILE_PATH}' not found. Run recognition first.")
        return None
    df = pd.read_csv(LOG_FILE_PATH)
    if df.empty:
        print("[INFO] Log file is empty — no charts to generate.")
        return None
    return df


# ---------------------------------------------------------------------------
# Chart 1: Attendance count per person
# ---------------------------------------------------------------------------
def chart_attendance_per_person(df: pd.DataFrame) -> str:
    counts = df["Name"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(8, max(4, len(counts) * 0.5)))
    bars = ax.barh(counts.index, counts.values)
    ax.set_xlabel("Attendance Count")
    ax.set_title("Attendance Count per Person")
    ax.bar_label(bars, padding=3, fmt="%d")
    plt.tight_layout()
    path = os.path.join(ANALYTICS_DIR, "chart_attendance_per_person.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"[CHART] Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Chart 2: Attendance over time (by date)
# ---------------------------------------------------------------------------
def chart_attendance_by_date(df: pd.DataFrame) -> str:
    if "Date" not in df.columns:
        print("[WARNING] No 'Date' column — skipping time chart.")
        return ""
    date_counts = df.groupby("Date").size().reset_index(name="Count")
    date_counts["Date"] = pd.to_datetime(date_counts["Date"])
    date_counts.sort_values("Date", inplace=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(date_counts["Date"], date_counts["Count"], marker="o", linewidth=2)
    ax.fill_between(date_counts["Date"], date_counts["Count"], alpha=0.15)
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Entries")
    ax.set_title("Attendance Trend Over Time")
    plt.xticks(rotation=30, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    path = os.path.join(ANALYTICS_DIR, "chart_attendance_by_date.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"[CHART] Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Chart 3: Known vs Unknown
# ---------------------------------------------------------------------------
def chart_known_vs_unknown(df: pd.DataFrame) -> str:
    known_count   = (df["Name"] != "Unknown").sum()
    unknown_count = (df["Name"] == "Unknown").sum()
    labels  = ["Known", "Unknown"]
    sizes   = [known_count, unknown_count]
    # Remove zero slices
    pairs   = [(l, s) for l, s in zip(labels, sizes) if s > 0]
    if not pairs:
        print("[WARNING] No data for Known/Unknown chart.")
        return ""
    labels, sizes = zip(*pairs)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
    ax.set_title("Known vs Unknown Face Detections")
    plt.tight_layout()
    path = os.path.join(ANALYTICS_DIR, "chart_known_vs_unknown.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"[CHART] Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def generate_analytics() -> list[str]:
    ensure_dir(ANALYTICS_DIR)
    df = _load_log()
    if df is None:
        return []

    saved = []
    saved.append(chart_attendance_per_person(df))
    saved.append(chart_attendance_by_date(df))
    saved.append(chart_known_vs_unknown(df))
    saved = [p for p in saved if p]

    print(f"\n[INFO] {len(saved)} chart(s) saved to '{ANALYTICS_DIR}/'.")
    return saved


if __name__ == "__main__":
    paths = generate_analytics()
    if not paths:
        sys.exit(1)
