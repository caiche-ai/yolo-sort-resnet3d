#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${ACTION_PYTHON:-/home/p/mambaforge/envs/python38/bin/python}"

make -C "${SCRIPT_DIR}"
export PYTHONPATH="/usr/lib/python3/dist-packages:/usr/lib/python3.8/dist-packages:/home/p/.local/lib/python3.8/site-packages${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON_BIN}" "${SCRIPT_DIR}/run_yolo_tracker_action.py" "$@"
