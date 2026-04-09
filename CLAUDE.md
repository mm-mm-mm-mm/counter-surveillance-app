# Counter-Surveillance App

## Project Overview

A macOS-native web application that processes video files to detect, track, and log motor vehicles for counter-surveillance purposes. The user interacts via a local web browser.

## Target Platform

- **Hardware:** Mac with Apple Silicon (M-series)
- **Acceleration:** Neural Engine and GPU cores for ML tasks (CoreML / Metal)
- **Interface:** Local web server accessed via browser

## Architecture

- **Backend:** Python (FastAPI or similar), handles video processing, ML inference, database
- **Frontend:** Web UI served locally (HTML/CSS/JS or lightweight framework)
- **Database:** SQLite for runtime, exported as CSV at session end
- **ML Pipeline:** Object detection (vehicle make/model/color) + ANPR (licence plate reading)

## Database Schema

| Field | Description |
|---|---|
| `observation_id` | `<video_filename>_<n>` (e.g. `clip001_1`) |
| `date_time_first_observation` | ISO timestamp from video metadata + elapsed time |
| `date_time_last_observation` | ISO timestamp when vehicle left frame |
| `vehicle_make` | Detected make |
| `vehicle_model` | Detected model |
| `vehicle_color` | Detected color |
| `vehicle_licence_plate` | ANPR result |
| `vehicle_licence_plate_color` | Background color of plate |
| `vehicle_licence_plate_nationality` | Detected nationality |
| `category` | `normal`, `taxi` (yellow plate), or `diplomatic` (blue plate) |

## Key Behaviors

- Timestamps are derived from **video file metadata** (creation time) plus elapsed playback time — not system clock
- Stationary vehicles at video start are logged with metadata start time; their `date_time_last_observation` is set when they move or the video ends
- Vehicles still in frame at video end keep their last observation box visible on screen
- Each detected vehicle gets a `.jpg` saved to `observation_images/<observation_id>.jpg`
- Session data is exported as `session_data/<video_filename>.csv` when video finishes

## Folder Structure

```
video_input/          # User places input video files here
observation_images/   # Auto-generated vehicle snapshots
session_data/         # Auto-generated CSV exports
```

## Commit Conventions

- Commit and push after each meaningful unit of work
- Use HTTPS for git remote (SSH keys not configured)
- Co-author commits with Claude: `Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>`
