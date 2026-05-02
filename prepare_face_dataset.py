#!/usr/bin/env python3
"""
Prepare student photo metadata for face-recognition attendance.

What this script does:
1) Reads source CSV with columns like:
   - Student ID
   - Student Name
   - Photo Link
2) Removes fully empty rows.
3) Normalizes column names and trims whitespace.
4) Converts Google Drive links to direct-download links.
5) Saves:
   - cleaned metadata CSV
   - summary CSV by student
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean and normalize student photo dataset CSV."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to source CSV (e.g., StudentPicsDataset.csv).",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory to write output CSV files.",
    )
    return parser.parse_args()


def _extract_drive_file_id(url: str) -> str | None:
    if not isinstance(url, str) or not url.strip():
        return None

    cleaned = url.strip()
    patterns = [
        r"id=([a-zA-Z0-9_-]+)",
        r"/d/([a-zA-Z0-9_-]+)",
        r"file/d/([a-zA-Z0-9_-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, cleaned)
        if match:
            return match.group(1)
    return None


def to_direct_drive_url(url: str) -> str:
    file_id = _extract_drive_file_id(url)
    if file_id:
        return f"https://drive.google.com/uc?export=download&id={file_id}"
    return url.strip() if isinstance(url, str) else ""


def normalize_student_id(value: str) -> str:
    raw = value.strip()
    # Fix common CSV float-style ids, e.g. "231014241.0" -> "231014241".
    if re.fullmatch(r"\d+\.0+", raw):
        return raw.split(".", maxsplit=1)[0]
    return raw


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {
        "Student ID": "student_id",
        "Student Name": "student_name",
        "Photo Link": "photo_link",
    }
    df = df.rename(columns=col_map)

    required = ["student_id", "student_name", "photo_link"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Expected columns like: {list(col_map.keys())}"
        )

    # Keep only needed columns in a stable order.
    df = df[required].copy()

    for col in required:
        df[col] = df[col].astype(str).str.strip()

    # Remove placeholder "nan" strings from pandas conversion.
    df = df.replace({"nan": "", "None": ""})
    df["student_id"] = df["student_id"].apply(normalize_student_id)
    return df


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    # Drop fully empty rows early.
    df = df.dropna(how="all")
    df = normalize_columns(df)

    # Remove rows where all core fields are empty after normalization.
    df = df[~((df["student_id"] == "") & (df["student_name"] == "") & (df["photo_link"] == ""))]

    # Convert Google Drive links to direct links.
    df["image_url"] = df["photo_link"].apply(to_direct_drive_url)

    # Count photos per student.
    photo_counts = (
        df.groupby("student_id", dropna=False)["image_url"]
        .count()
        .reset_index(name="num_photos")
    )

    # Deduplicate exact duplicate rows while keeping multi-image rows.
    cleaned = df.drop_duplicates(
        subset=["student_id", "student_name", "image_url"]
    ).copy()

    # Recompute counts post dedupe.
    summary = (
        cleaned.groupby(["student_id", "student_name"], dropna=False)
        .size()
        .reset_index(name="num_unique_photos")
        .sort_values(["num_unique_photos", "student_id"], ascending=[False, True])
    )

    # Save output files.
    cleaned_out = output_dir / "student_photos_clean.csv"
    summary_out = output_dir / "student_photo_summary.csv"
    counts_out = output_dir / "student_photo_counts_raw.csv"

    cleaned.to_csv(cleaned_out, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_out, index=False, encoding="utf-8-sig")
    photo_counts.to_csv(counts_out, index=False, encoding="utf-8-sig")

    print("Done.")
    print(f"Input rows (after dropna all): {len(df)}")
    print(f"Clean unique rows: {len(cleaned)}")
    print(f"Unique students: {cleaned['student_id'].nunique()}")
    print(f"Wrote: {cleaned_out}")
    print(f"Wrote: {summary_out}")
    print(f"Wrote: {counts_out}")


if __name__ == "__main__":
    main()
