#!/bin/sh
# Regenerate index.html after editing page.json.
# Run from any directory (copy the command below):
# /Users/chriscremer/Downloads/soccer_video_analysis_storage/processed_matches/CTS_SEP22_Dream_Team_Vs_Ballers_FC_run/CTS_SEP22_Dream_Team_Vs_Ballers_FC/render_timeline.sh
set -eu
page_dir=$(CDPATH= cd "$(dirname "$0")" && pwd)
exec /Users/chriscremer/Downloads/soccer_video_analysis_storage/.venv/bin/python /Users/chriscremer/code/SVA/SVA/m_soccer_video_analysis/soccer_minutes/build_game_page.py --config "$page_dir/page.json"
