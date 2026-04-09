#!/bin/bash
set -e
cd "$(dirname "$0")"
source cs_app_venv/bin/activate
python -m uvicorn cs_app.main:app --host 127.0.0.1 --port 8000 --reload
