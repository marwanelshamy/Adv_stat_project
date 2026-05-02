# AI Classroom Monitoring Project

This project provides an end-to-end workflow for classroom monitoring:

- Doctor login and lecture selection in a `Shiny` UI
- Start attendance session from camera/video/image
- Face recognition attendance export
- Session statuses (`present`, `late`, `left_early`, `absent`)
- Session-level summary and statistics files

## Project Structure

- `prepare_face_dataset.py` - cleans student photo metadata CSV
- `download_student_images.py` - downloads images to `faces/<student_id>/`
- `build_face_embeddings.py` - builds `output/face_embeddings.npz`
- `take_attendance_realtime.py` - runs attendance session and writes CSV outputs
- `app_shiny.R` - dashboard with login, lecture selection, start session, statistics
- `config/doctors.csv` - doctor login credentials and subjects
- `config/lectures.csv` - lecture schedule

## 1) Python Setup

```powershell
cd "C:\Users\Hero\Documents\sem6\adv.stat\Project"
python -m pip install -r requirements.txt
```

## 2) Prepare Face Data (already done in your environment)

```powershell
python ".\prepare_face_dataset.py" --input "C:\Users\Hero\Documents\sem6\adv.stat\StudentPicsDataset.csv" --output-dir ".\output"
python ".\download_student_images.py" --input ".\output\student_photos_clean.csv" --output-dir ".\faces"
python ".\build_face_embeddings.py" --faces-dir ".\faces" --output-dir ".\output"
```

## 3) Run Attendance from CLI

```powershell
python ".\take_attendance_realtime.py" `
  --embeddings ".\output\face_embeddings.npz" `
  --source 0 `
  --output-dir ".\output\sessions" `
  --session-id "lecture_demo" `
  --tolerance 0.65 `
  --frame-skip 1 `
  --show-window `
  --late-threshold-minutes 10 `
  --left-early-gap-minutes 10
```

Outputs:

- `output/sessions/<session_id>_attendance.csv`
- `output/sessions/<session_id>_events.csv`
- `output/sessions/<session_id>_summary.csv`

## 4) Run Shiny Dashboard

Option A:

```powershell
Rscript -e "shiny::runApp('app_shiny.R', launch.browser = TRUE)"
```

Option B:

- Double-click `run_shiny.bat`

## Demo Credentials

From `config/doctors.csv`:

- Username: `ahmed` / Password: `1234`
- Username: `mona` / Password: `1234`

## Notes

- For Camo camera, change camera index in the dashboard/CLI (`0`, `1`, `2`, ...).
- If recognition is strict, increase tolerance (`0.65` to `0.70`).
- `face_recognition` currently uses `setuptools<81` compatibility path.
