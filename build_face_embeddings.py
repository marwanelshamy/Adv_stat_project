#!/usr/bin/env python3
"""
Build a face-embedding database from student image folders.

Expected input structure:
faces/
  <student_id>/
    img_001.jpg
    img_002.jpg

Outputs:
- output/face_embeddings.csv
- output/face_embeddings.npz
- output/face_embedding_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build face embeddings per student from downloaded images."
    )
    parser.add_argument(
        "--faces-dir",
        default="faces",
        help="Root folder containing student face images.",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory to write embedding outputs.",
    )
    parser.add_argument(
        "--detection-model",
        default="hog",
        choices=["hog", "cnn"],
        help="face_recognition model for detection (hog faster, cnn more accurate).",
    )
    parser.add_argument(
        "--num-jitters",
        type=int,
        default=1,
        help="How many times to re-sample face when encoding (higher = slower).",
    )
    parser.add_argument(
        "--allow-multi-face",
        action="store_true",
        help="Keep images that contain multiple faces (default: skip them).",
    )
    return parser.parse_args()


def iter_image_files(folder: Path) -> Iterable[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def _import_face_recognition():
    try:
        import face_recognition  # type: ignore

        return face_recognition
    except Exception as exc:
        raise RuntimeError(
            "face_recognition package is required.\n"
            "Install with: pip install face-recognition\n"
            f"Original import error: {exc}"
        ) from exc


def normalize_student_id(value: str) -> str:
    raw = value.strip()
    if re.fullmatch(r"\d+\.0+", raw):
        return raw.split(".", maxsplit=1)[0]
    return raw


def main() -> None:
    args = parse_args()
    faces_dir = Path(args.faces_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not faces_dir.exists():
        raise FileNotFoundError(f"Faces directory not found: {faces_dir}")

    face_recognition = _import_face_recognition()

    student_dirs = sorted([p for p in faces_dir.iterdir() if p.is_dir()])
    if not student_dirs:
        raise ValueError(f"No student subfolders found in: {faces_dir}")

    rows: list[dict[str, object]] = []
    failed_images: list[dict[str, str]] = []
    total_images = 0
    encoded_images = 0

    for student_dir in student_dirs:
        student_id = normalize_student_id(student_dir.name.strip())
        image_paths = list(iter_image_files(student_dir))

        for image_path in image_paths:
            total_images += 1
            try:
                image = face_recognition.load_image_file(str(image_path))
                locations = face_recognition.face_locations(
                    image, model=args.detection_model
                )
                if len(locations) == 0:
                    failed_images.append(
                        {
                            "student_id": student_id,
                            "image_path": str(image_path),
                            "reason": "no_face_detected",
                        }
                    )
                    continue
                if len(locations) > 1 and not args.allow_multi_face:
                    failed_images.append(
                        {
                            "student_id": student_id,
                            "image_path": str(image_path),
                            "reason": "multiple_faces_detected",
                        }
                    )
                    continue

                encodings = face_recognition.face_encodings(
                    image,
                    known_face_locations=locations,
                    num_jitters=args.num_jitters,
                )
                if not encodings:
                    failed_images.append(
                        {
                            "student_id": student_id,
                            "image_path": str(image_path),
                            "reason": "encoding_failed",
                        }
                    )
                    continue

                for face_idx, emb in enumerate(encodings, start=1):
                    rows.append(
                        {
                            "student_id": student_id,
                            "image_path": str(image_path),
                            "face_index": face_idx,
                            "embedding": emb.astype(np.float32).tolist(),
                        }
                    )
                encoded_images += 1
            except Exception as exc:
                failed_images.append(
                    {
                        "student_id": student_id,
                        "image_path": str(image_path),
                        "reason": f"error:{exc}",
                    }
                )

    if not rows:
        raise RuntimeError(
            "No embeddings generated. Check image quality and face detection settings."
        )

    ids = np.array([r["student_id"] for r in rows], dtype=object)
    img_paths = np.array([r["image_path"] for r in rows], dtype=object)
    embs = np.array([r["embedding"] for r in rows], dtype=np.float32)

    npz_path = output_dir / "face_embeddings.npz"
    np.savez_compressed(npz_path, student_ids=ids, image_paths=img_paths, embeddings=embs)

    csv_path = output_dir / "face_embeddings.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["student_id", "image_path", "face_index", "embedding_json"])
        for r in rows:
            writer.writerow(
                [r["student_id"], r["image_path"], r["face_index"], json.dumps(r["embedding"])]
            )

    summary: dict[str, int] = {}
    for sid in ids:
        summary[str(sid)] = summary.get(str(sid), 0) + 1

    summary_path = output_dir / "face_embedding_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["student_id", "num_embeddings"])
        for sid, count in sorted(summary.items(), key=lambda x: x[0]):
            writer.writerow([sid, count])

    fail_log = output_dir / "face_embedding_failures.csv"
    with fail_log.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["student_id", "image_path", "reason"])
        writer.writeheader()
        for row in failed_images:
            writer.writerow(row)

    print("Embedding build complete.")
    print(f"Faces directory: {faces_dir}")
    print(f"Student folders: {len(student_dirs)}")
    print(f"Total images seen: {total_images}")
    print(f"Images encoded: {encoded_images}")
    print(f"Embedding rows: {len(rows)}")
    print(f"Failed images: {len(failed_images)}")
    print(f"Wrote: {npz_path}")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {fail_log}")


if __name__ == "__main__":
    main()
