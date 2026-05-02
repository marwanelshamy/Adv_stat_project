#!/usr/bin/env python3
"""
Realtime attendance from webcam/video/image using prebuilt face embeddings.

Inputs:
- output/face_embeddings.npz
- source: webcam index, video path, or image path

Outputs:
- output/sessions/<session_id>_attendance.csv
- output/sessions/<session_id>_events.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}


@dataclass
class SeenInfo:
    first_seen: str
    last_seen: str
    sightings: int
    best_distance: float
    emotion_counts: Dict[str, int] = field(default_factory=dict)
    emotion_conf_sum: float = 0.0
    emotion_samples: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Take attendance from webcam/video/image using face embeddings."
    )
    parser.add_argument(
        "--embeddings",
        default="output/face_embeddings.npz",
        help="Path to face embeddings npz file.",
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Video source: webcam index (e.g. 0), video path, or image path.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/sessions",
        help="Directory where session CSV files are saved.",
    )
    parser.add_argument(
        "--session-id",
        default="",
        help="Optional session id. If empty, generated automatically.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.45,
        help="Face-match distance threshold (lower = stricter).",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=2,
        help="Process every Nth frame for speed.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional frame limit for tests (0 = unlimited).",
    )
    parser.add_argument(
        "--show-window",
        action="store_true",
        help="Show live preview window with detections.",
    )
    parser.add_argument(
        "--save-unknown",
        action="store_true",
        help="Save unknown face crops under output/sessions/<session_id>_unknowns.",
    )
    parser.add_argument(
        "--camera-backend",
        default="auto",
        choices=["auto", "dshow", "msmf"],
        help="Camera backend on Windows. Try dshow for virtual cameras like Camo.",
    )
    parser.add_argument(
        "--enable-emotion",
        action="store_true",
        help="Enable emotion detection.",
    )
    parser.add_argument(
        "--emotion-engine",
        default="heuristic",
        choices=["heuristic", "deepface", "auto"],
        help="Emotion engine. 'heuristic' works without extra ML dependencies.",
    )
    parser.add_argument(
        "--emotion-min-face-size",
        type=int,
        default=48,
        help="Minimum detected face width/height for emotion inference.",
    )
    parser.add_argument(
        "--late-threshold-minutes",
        type=float,
        default=10.0,
        help="Late threshold in minutes from session start.",
    )
    parser.add_argument(
        "--left-early-gap-minutes",
        type=float,
        default=10.0,
        help="If last seen earlier than (session end - gap), mark left_early.",
    )
    return parser.parse_args()


def _import_runtime_modules():
    try:
        import cv2  # type: ignore
        import face_recognition  # type: ignore

        return cv2, face_recognition
    except Exception as exc:
        raise RuntimeError(
            "Missing runtime dependencies.\n"
            "Install with:\n"
            "  pip install opencv-python face-recognition\n"
            f"Original error: {exc}"
        ) from exc


def _import_deepface_if_enabled(enable_emotion: bool, emotion_engine: str):
    if not enable_emotion or emotion_engine == "heuristic":
        return None
    try:
        from deepface import DeepFace  # type: ignore

        return DeepFace
    except Exception as exc:
        if emotion_engine == "auto":
            return None
        raise RuntimeError(
            "Emotion mode requires DeepFace.\n"
            "Install with:\n"
            "  python -m pip install deepface\n"
            f"Original error: {exc}"
        ) from exc


def _load_embeddings(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    student_ids = data["student_ids"]
    embeddings = data["embeddings"].astype(np.float32)
    if len(student_ids) == 0 or len(embeddings) == 0:
        raise ValueError("Embeddings file is empty.")
    if len(student_ids) != len(embeddings):
        raise ValueError("Embeddings file is invalid (length mismatch).")
    return student_ids, embeddings


def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _make_session_id(user_session_id: str) -> str:
    if user_session_id.strip():
        return user_session_id.strip()
    return datetime.now().strftime("session_%Y%m%d_%H%M%S")


def _parse_ts(ts: str) -> datetime:
    return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")


def _parse_source(raw: str):
    raw = raw.strip()
    if raw.isdigit():
        return int(raw), "camera"
    p = Path(raw).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Source not found: {p}")
    if p.suffix.lower() in IMAGE_EXTS:
        return p, "image"
    return p, "video"


def _match_student(
    face_encoding: np.ndarray,
    known_student_ids: np.ndarray,
    known_embeddings: np.ndarray,
    tolerance: float,
) -> Tuple[str, float]:
    distances = np.linalg.norm(known_embeddings - face_encoding, axis=1)
    best_idx = int(np.argmin(distances))
    best_distance = float(distances[best_idx])
    if best_distance <= tolerance:
        return str(known_student_ids[best_idx]), best_distance
    return "UNKNOWN", best_distance


def _infer_emotion(deepface_module, face_bgr: np.ndarray) -> Tuple[str, float]:
    """
    Returns (dominant_emotion, confidence_percent).
    """
    if deepface_module is None:
        return "", 0.0
    try:
        result = deepface_module.analyze(
            img_path=face_bgr,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend="opencv",
            silent=True,
        )
        if isinstance(result, list):
            result = result[0]
        dominant = str(result.get("dominant_emotion", "")).strip().lower()
        emotions = result.get("emotion", {}) or {}
        conf = float(emotions.get(dominant, 0.0))
        return dominant, conf
    except Exception:
        return "", 0.0


def _infer_emotion_heuristic(
    cv2_module,
    face_bgr: np.ndarray,
    smile_cascade,
    eye_cascade,
) -> Tuple[str, float]:
    if face_bgr.size == 0:
        return "", 0.0

    gray = cv2_module.cvtColor(face_bgr, cv2_module.COLOR_BGR2GRAY)
    gray = cv2_module.equalizeHist(gray)

    smiles = smile_cascade.detectMultiScale(
        gray, scaleFactor=1.8, minNeighbors=20, minSize=(20, 20)
    )
    eyes = eye_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=6, minSize=(12, 12)
    )
    variance = float(cv2_module.Laplacian(gray, cv2_module.CV_64F).var())

    if len(smiles) > 0:
        return "happy", 80.0
    if len(eyes) == 0:
        return "bored", 62.0
    if variance < 45.0:
        return "confused", 58.0
    return "neutral", 60.0


def _write_outputs(
    out_dir: Path,
    session_id: str,
    all_student_ids: Set[str],
    seen: Dict[str, SeenInfo],
    events: List[dict],
    session_start: datetime,
    session_end: datetime,
    late_threshold_minutes: float,
    left_early_gap_minutes: float,
) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    attendance_path = out_dir / f"{session_id}_attendance.csv"
    events_path = out_dir / f"{session_id}_events.csv"

    with attendance_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "session_id",
                "student_id",
                "status",
                "is_late",
                "left_early",
                "first_seen",
                "last_seen",
                "sightings",
                "visibility_ratio",
                "engagement_score",
                "attitude_label",
                "dominant_emotion",
                "avg_emotion_confidence",
                "best_distance",
            ]
        )
        late_cutoff = session_start + timedelta(minutes=late_threshold_minutes)
        early_cutoff = session_end - timedelta(minutes=left_early_gap_minutes)

        for student_id in sorted(all_student_ids):
            if student_id not in seen:
                writer.writerow(
                    [
                        session_id,
                        student_id,
                        "absent",
                        0,
                        0,
                        "",
                        "",
                        0,
                        0,
                        0,
                        "at_risk",
                        "",
                        "",
                        "",
                    ]
                )
                continue

            info = seen[student_id]
            first_seen_dt = _parse_ts(info.first_seen)
            last_seen_dt = _parse_ts(info.last_seen)

            is_late = int(first_seen_dt > late_cutoff)
            left_early = int(last_seen_dt < early_cutoff)
            if is_late:
                status = "late"
            elif left_early:
                status = "left_early"
            else:
                status = "present"

            # Visibility ratio approximates attendance continuity during session.
            visibility_ratio = min(1.0, info.sightings / max(1, len(events)))
            engagement_score = 100.0
            if status == "late":
                engagement_score -= 15.0
            if status == "left_early":
                engagement_score -= 20.0
            # Penalize weak presence frequency.
            engagement_score -= max(0.0, (1.0 - visibility_ratio) * 35.0)
            # Slight confidence-based penalty when face distance is weak.
            engagement_score -= min(10.0, max(0.0, (info.best_distance - 0.40) * 25.0))
            engagement_score = max(0.0, min(100.0, engagement_score))

            if engagement_score >= 75:
                attitude_label = "engaged"
            elif engagement_score >= 50:
                attitude_label = "normal"
            else:
                attitude_label = "at_risk"

            dominant_emotion = ""
            avg_emotion_conf = ""
            if info.emotion_samples > 0:
                if info.emotion_counts:
                    dominant_emotion = Counter(info.emotion_counts).most_common(1)[0][0]
                avg_emotion_conf = round(info.emotion_conf_sum / info.emotion_samples, 2)

            writer.writerow(
                [
                    session_id,
                    student_id,
                    status,
                    is_late,
                    left_early,
                    info.first_seen,
                    info.last_seen,
                    info.sightings,
                    round(visibility_ratio, 4),
                    round(engagement_score, 2),
                    attitude_label,
                    dominant_emotion,
                    avg_emotion_conf,
                    round(info.best_distance, 6),
                ]
            )

    with events_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "session_id",
                "timestamp",
                "frame_index",
                "student_id",
                "distance",
                "emotion",
                "emotion_confidence",
                "top",
                "right",
                "bottom",
                "left",
            ],
        )
        writer.writeheader()
        for row in events:
            writer.writerow(row)

    return attendance_path, events_path


def _write_session_summary(
    out_dir: Path,
    session_id: str,
    attendance_path: Path,
) -> Path:
    summary_path = out_dir / f"{session_id}_summary.csv"
    status_counts = {"present": 0, "late": 0, "left_early": 0, "absent": 0}
    attitude_counts = {"engaged": 0, "normal": 0, "at_risk": 0}
    emotion_counts: Dict[str, int] = {}
    total_students = 0
    score_sum = 0.0
    score_n = 0

    with attendance_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_students += 1
            status = (row.get("status") or "").strip()
            if status in status_counts:
                status_counts[status] += 1
            attitude = (row.get("attitude_label") or "").strip()
            if attitude in attitude_counts:
                attitude_counts[attitude] += 1
            raw_score = (row.get("engagement_score") or "").strip()
            if raw_score != "":
                try:
                    score_sum += float(raw_score)
                    score_n += 1
                except ValueError:
                    pass
            em = (row.get("dominant_emotion") or "").strip().lower()
            if em:
                emotion_counts[em] = emotion_counts.get(em, 0) + 1

    present_like = (
        status_counts["present"] + status_counts["late"] + status_counts["left_early"]
    )
    attendance_rate = (present_like / total_students * 100.0) if total_students > 0 else 0.0
    avg_engagement = (score_sum / score_n) if score_n > 0 else 0.0

    with summary_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "session_id",
                "total_students",
                "present_count",
                "late_count",
                "left_early_count",
                "absent_count",
                "attendance_rate_percent",
                "engaged_count",
                "normal_count",
                "at_risk_count",
                "avg_engagement_score",
                "top_emotion",
                "top_emotion_count",
            ]
        )
        top_emotion = ""
        top_emotion_count = 0
        if emotion_counts:
            top_emotion, top_emotion_count = sorted(
                emotion_counts.items(), key=lambda kv: kv[1], reverse=True
            )[0]
        writer.writerow(
            [
                session_id,
                total_students,
                status_counts["present"],
                status_counts["late"],
                status_counts["left_early"],
                status_counts["absent"],
                round(attendance_rate, 2),
                attitude_counts["engaged"],
                attitude_counts["normal"],
                attitude_counts["at_risk"],
                round(avg_engagement, 2),
                top_emotion,
                top_emotion_count,
            ]
        )

    return summary_path


def main() -> None:
    args = parse_args()
    cv2, face_recognition = _import_runtime_modules()
    DeepFace = _import_deepface_if_enabled(args.enable_emotion, args.emotion_engine)

    smile_cascade = None
    eye_cascade = None
    if args.enable_emotion and (args.emotion_engine in {"heuristic", "auto"}):
        smile_xml = cv2.data.haarcascades + "haarcascade_smile.xml"
        eye_xml = cv2.data.haarcascades + "haarcascade_eye.xml"
        smile_cascade = cv2.CascadeClassifier(smile_xml)
        eye_cascade = cv2.CascadeClassifier(eye_xml)

    session_id = _make_session_id(args.session_id)
    embeddings_path = Path(args.embeddings).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    source, source_type = _parse_source(args.source)

    known_student_ids, known_embeddings = _load_embeddings(embeddings_path)
    all_student_ids = {str(sid) for sid in known_student_ids}
    session_start_dt = datetime.now()

    seen: Dict[str, SeenInfo] = {}
    events: List[dict] = []
    frame_index = 0
    unknown_count = 0

    unknown_dir = output_dir / f"{session_id}_unknowns"
    if args.save_unknown:
        unknown_dir.mkdir(parents=True, exist_ok=True)

    def process_frame(frame_bgr):
        nonlocal unknown_count, frame_index
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb, model="hog")
        encodings = face_recognition.face_encodings(rgb, known_face_locations=locations)

        for (top, right, bottom, left), enc in zip(locations, encodings):
            face_w = max(0, int(right - left))
            face_h = max(0, int(bottom - top))
            face_crop = frame_bgr[max(0, top) : max(0, bottom), max(0, left) : max(0, right)]
            emotion_label = ""
            emotion_conf = 0.0
            if (
                args.enable_emotion
                and face_crop.size > 0
                and face_w >= args.emotion_min_face_size
                and face_h >= args.emotion_min_face_size
            ):
                if DeepFace is not None:
                    emotion_label, emotion_conf = _infer_emotion(DeepFace, face_crop)
                elif smile_cascade is not None and eye_cascade is not None:
                    emotion_label, emotion_conf = _infer_emotion_heuristic(
                        cv2, face_crop, smile_cascade, eye_cascade
                    )

            student_id, dist = _match_student(
                np.asarray(enc, dtype=np.float32),
                known_student_ids,
                known_embeddings,
                args.tolerance,
            )
            ts = _now_str()
            events.append(
                {
                    "session_id": session_id,
                    "timestamp": ts,
                    "frame_index": frame_index,
                    "student_id": student_id,
                    "distance": round(dist, 6),
                    "emotion": emotion_label,
                    "emotion_confidence": round(emotion_conf, 2),
                    "top": int(top),
                    "right": int(right),
                    "bottom": int(bottom),
                    "left": int(left),
                }
            )

            if student_id != "UNKNOWN":
                if student_id not in seen:
                    seen[student_id] = SeenInfo(
                        first_seen=ts,
                        last_seen=ts,
                        sightings=1,
                        best_distance=dist,
                    )
                else:
                    info = seen[student_id]
                    info.last_seen = ts
                    info.sightings += 1
                    info.best_distance = min(info.best_distance, dist)
                if emotion_label:
                    info = seen[student_id]
                    info.emotion_counts[emotion_label] = (
                        info.emotion_counts.get(emotion_label, 0) + 1
                    )
                    info.emotion_conf_sum += emotion_conf
                    info.emotion_samples += 1
            elif args.save_unknown:
                unknown_count += 1
                crop = frame_bgr[max(0, top) : max(0, bottom), max(0, left) : max(0, right)]
                if crop.size > 0:
                    crop_path = unknown_dir / f"unknown_{frame_index:06d}_{unknown_count:03d}.jpg"
                    cv2.imwrite(str(crop_path), crop)

            if args.show_window:
                color = (0, 200, 0) if student_id != "UNKNOWN" else (0, 0, 255)
                cv2.rectangle(frame_bgr, (left, top), (right, bottom), color, 2)
                label = f"{student_id} ({dist:.3f})"
                cv2.putText(
                    frame_bgr,
                    label,
                    (left, max(15, top - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )

        return frame_bgr

    if source_type == "image":
        frame = cv2.imread(str(source))
        if frame is None:
            raise RuntimeError(f"Unable to read image: {source}")
        process_frame(frame)
        frame_index = 1
        if args.show_window:
            cv2.imshow("Attendance", frame)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()
    else:
        if source_type == "camera":
            if args.camera_backend == "dshow":
                cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            elif args.camera_backend == "msmf":
                cap = cv2.VideoCapture(source, cv2.CAP_MSMF)
            else:
                cap = cv2.VideoCapture(source)
        else:
            cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open source: {source}. "
                "Try another camera index (--source 1/2/3) or --camera-backend dshow."
            )
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_index += 1
                if args.frame_skip > 1 and (frame_index % args.frame_skip != 0):
                    if args.show_window:
                        cv2.imshow("Attendance", frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                    continue

                annotated = process_frame(frame)

                if args.show_window:
                    cv2.imshow("Attendance", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                if args.max_frames > 0 and frame_index >= args.max_frames:
                    break
        finally:
            cap.release()
            if args.show_window:
                cv2.destroyAllWindows()

    session_end_dt = datetime.now()
    attendance_path, events_path = _write_outputs(
        output_dir,
        session_id,
        all_student_ids,
        seen,
        events,
        session_start_dt,
        session_end_dt,
        args.late_threshold_minutes,
        args.left_early_gap_minutes,
    )
    summary_path = _write_session_summary(output_dir, session_id, attendance_path)

    print("Session complete.")
    print(f"Session ID: {session_id}")
    print(f"Source type: {source_type}")
    print(f"Frames read: {frame_index}")
    print(f"Recognized students: {len(seen)}")
    print(f"Total detection events: {len(events)}")
    print(f"Attendance CSV: {attendance_path}")
    print(f"Events CSV: {events_path}")
    print(f"Summary CSV: {summary_path}")
    if args.save_unknown:
        print(f"Unknown faces folder: {unknown_dir}")


if __name__ == "__main__":
    main()
