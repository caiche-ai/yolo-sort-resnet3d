#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
make -C "${script_dir}"
exec /usr/bin/python3 "${script_dir}/run_yolo_tracker.py" "$@"
