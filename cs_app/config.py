import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

VIDEO_INPUT_DIR = BASE_DIR / "video_input"
OBSERVATION_IMAGES_DIR = BASE_DIR / "observation_images"
SESSION_DATA_DIR = BASE_DIR / "session_data"
MODELS_DIR = BASE_DIR / "models"
FRONTEND_DIR = BASE_DIR / "cs_app" / "frontend"

# ML device: "mps" for Apple Silicon GPU, "cpu" as fallback
DEVICE = "mps"

# Process every Nth frame (1 = every frame, 2 = every other frame, etc.)
FRAME_SKIP = 2

# Target WebSocket frame send rate (frames per second)
TARGET_FPS = 15

# Minimum ANPR confidence to mark plate as confirmed
ANPR_CONFIDENCE_THRESHOLD = 0.85

# Number of frames a track must be absent before declared departed
TRACK_GRACE_FRAMES = 10

# Frames at the start of video within which a vehicle is considered stationary
STATIONARY_START_FRAMES = 30

# JPEG encode quality for frame streaming (0-100)
FRAME_JPEG_QUALITY = 75

DATABASE_URL = f"sqlite+aiosqlite:///{SESSION_DATA_DIR}/runtime.db"
