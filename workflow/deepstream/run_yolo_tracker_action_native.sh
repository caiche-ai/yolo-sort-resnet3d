#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

make -C "${SCRIPT_DIR}"
if [[ ! -e "${SCRIPT_DIR}/libnvds_custom_sequence_track_id.so" ]]; then
  bash "${SCRIPT_DIR}/build_track_sequence_preprocess.sh"
fi
exec /usr/bin/python3 "${SCRIPT_DIR}/run_yolo_tracker_action_native.py" "$@"
