#!/bin/sh
set -eu
PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
DATA_DIR=${PICKLEBALL_DATA_DIR:-/Users/chriscremer/Downloads/pickleball_video_analysis}
PYTHON_BIN=${PICKLEBALL_PYTHON:-$DATA_DIR/.venv/bin/python}
if [ ! -x "$PYTHON_BIN" ]; then PYTHON_BIN=python3; fi
exec "$PYTHON_BIN" "$PROJECT_DIR/server.py" --data "$DATA_DIR" "$@"
