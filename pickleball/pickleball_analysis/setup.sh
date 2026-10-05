#!/bin/sh
set -eu
PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
DATA_DIR=${PICKLEBALL_DATA_DIR:-/Users/chriscremer/Downloads/pickleball_video_analysis}
mkdir -p "$DATA_DIR"
if [ ! -x "$DATA_DIR/.venv/bin/python" ]; then
  "${PICKLEBALL_PYTHON:-python3}" -m venv "$DATA_DIR/.venv"
fi
"$DATA_DIR/.venv/bin/python" -m pip install -r "$PROJECT_DIR/requirements.txt"
