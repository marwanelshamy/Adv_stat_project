#!/usr/bin/env python3
"""
Download student images from cleaned CSV metadata.

Expected CSV columns:
- student_id
- student_name
- image_url

Output structure:
faces/
  <student_id>/
    img_001.jpg
    img_002.jpg
"""

from __future__ import annotations

import argparse
import csv
import re
import ssl
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download student images into per-student folders."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to cleaned CSV (e.g., output/student_photos_clean.csv).",
    )
    parser.add_argument(
        "--output-dir",
        default="faces",
        help="Output root directory for downloaded images.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=20,
        help="HTTP timeout seconds per image (default: 20).",
    )
    return parser.parse_args()


def _safe_ext_from_url(url: str) -> str:
    path = urllib.parse.urlparse(url).path.lower()
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
        if path.endswith(ext):
            return ext
    return ".jpg"


def normalize_student_id(value: str) -> str:
    raw = value.strip()
    if re.fullmatch(r"\d+\.0+", raw):
        return raw.split(".", maxsplit=1)[0]
    return raw


def _guess_extension_from_bytes(content: bytes, fallback: str = ".jpg") -> str:
    # Basic magic-byte checks to avoid extra dependencies.
    if content.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if content.startswith(b"GIF87a") or content.startswith(b"GIF89a"):
        return ".gif"
    if content.startswith(b"RIFF") and content[8:12] == b"WEBP":
        return ".webp"
    if content.startswith(b"BM"):
        return ".bmp"
    return fallback


def _download_bytes(url: str, timeout: int) -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        },
    )
    context = ssl.create_default_context()
    with urllib.request.urlopen(req, timeout=timeout, context=context) as response:
        return response.read()


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with input_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    required = {"student_id", "student_name", "image_url"}
    if not rows:
        raise ValueError("Input CSV is empty.")
    if not required.issubset(rows[0].keys()):
        raise ValueError(
            f"Missing required columns. Expected: {sorted(required)}; "
            f"Found: {sorted(rows[0].keys())}"
        )

    # Track image index per student for stable file names.
    counters: dict[str, int] = {}
    success = 0
    failed = 0
    log_lines = []

    for row in rows:
        student_id = normalize_student_id((row.get("student_id") or "").strip())
        image_url = (row.get("image_url") or "").strip()
        if not student_id or not image_url:
            failed += 1
            log_lines.append(f"SKIP missing data: student_id={student_id}, url={image_url}")
            continue

        counters[student_id] = counters.get(student_id, 0) + 1
        idx = counters[student_id]

        student_dir = output_dir / student_id
        student_dir.mkdir(parents=True, exist_ok=True)

        fallback_ext = _safe_ext_from_url(image_url)
        base_name = f"img_{idx:03d}"
        temp_path = student_dir / f"{base_name}{fallback_ext}"

        try:
            content = _download_bytes(image_url, timeout=args.timeout)
            if not content:
                raise ValueError("Empty response body")

            ext = _guess_extension_from_bytes(content, fallback=fallback_ext)
            final_path = student_dir / f"{base_name}{ext}"
            final_path.write_bytes(content)
            success += 1
        except (urllib.error.URLError, urllib.error.HTTPError, ValueError, TimeoutError) as exc:
            failed += 1
            log_lines.append(f"FAIL student_id={student_id} url={image_url} error={exc}")

    log_path = output_dir / "download_log.txt"
    log_path.write_text("\n".join(log_lines), encoding="utf-8")

    print("Download finished.")
    print(f"Total rows processed: {len(rows)}")
    print(f"Successful downloads: {success}")
    print(f"Failed downloads: {failed}")
    print(f"Student folders created: {len(counters)}")
    print(f"Log file: {log_path}")


if __name__ == "__main__":
    main()
