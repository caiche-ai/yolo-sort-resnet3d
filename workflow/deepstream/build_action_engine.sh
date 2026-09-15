#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${ACTION_PYTHON:-/home/p/mambaforge/envs/python38/bin/python}"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
ONNX_PATH="${PROJECT_ROOT}/workflow/resource/action_resnet3d_t30_s200.onnx"
ENGINE_PATH="${PROJECT_ROOT}/workflow/resource/action_resnet3d_t30_s200.fp16.engine"

if [[ -e "${ONNX_PATH}" || -e "${ENGINE_PATH}" ]]; then
  echo "Refusing to overwrite ${ONNX_PATH} or ${ENGINE_PATH}" >&2
  exit 1
fi

"${PYTHON_BIN}" "${PROJECT_ROOT}/workflow/itfact/export_action_onnx.py" \
  --clip-length 30 \
  --input-size 200 \
  --output "${ONNX_PATH}"

"${TRTEXEC}" \
  --onnx="${ONNX_PATH}" \
  --saveEngine="${ENGINE_PATH}" \
  --fp16 \
  --minShapes=normalized_clips:1x3x30x200x200 \
  --optShapes=normalized_clips:2x3x30x200x200 \
  --maxShapes=normalized_clips:4x3x30x200x200 \
  --workspace=3072 \
  --buildOnly
