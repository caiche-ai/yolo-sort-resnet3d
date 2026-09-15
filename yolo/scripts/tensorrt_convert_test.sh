#!/usr/bin/env bash

set -Eeuo pipefail

usage() {
    cat <<'EOF'
Convert an ONNX model to a fixed-shape FP16 TensorRT engine and benchmark it.

Usage:
  bash yolo/scripts/tensorrt_convert_test.sh [MODEL] [IMAGE_SIZE] [BATCH_SIZE] [INPUT_NAME]

Defaults:
  MODEL       yolo/edge_pt/onnx/worker.640.yolov5n6.onnx
  IMAGE_SIZE  640
  BATCH_SIZE  1
  INPUT_NAME  images

Environment variables:
  TRTEXEC        trtexec path (default: /usr/src/tensorrt/bin/trtexec)
  WORKSPACE_MIB  TensorRT workspace in MiB (default: 1024)
  WARMUP_MS      Benchmark warm-up in milliseconds (default: 1000)
  DURATION_S     Benchmark duration in seconds (default: 10)
  ENGINE_PATH    Output engine path (default: MODEL with .fp16.engine suffix)
  REBUILD        Set to 1 to rebuild an existing engine (default: 0)
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
cd "${repo_root}"

model_path="${1:-yolo/edge_pt/onnx/worker.640.yolov5n6.onnx}"
image_size="${2:-640}"
batch_size="${3:-1}"
input_name="${4:-images}"

trtexec="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
workspace_mib="${WORKSPACE_MIB:-1024}"
warmup_ms="${WARMUP_MS:-1000}"
duration_s="${DURATION_S:-10}"
rebuild="${REBUILD:-0}"
engine_path="${ENGINE_PATH:-${model_path%.onnx}.fp16.engine}"

for value_name in image_size batch_size workspace_mib warmup_ms duration_s; do
    value="${!value_name}"
    if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
        printf 'error: %s must be a positive integer, got %s\n' "${value_name}" "${value}" >&2
        exit 2
    fi
done

if [[ ! -x "${trtexec}" ]]; then
    printf 'error: trtexec is not executable: %s\n' "${trtexec}" >&2
    exit 2
fi

if [[ ! -f "${model_path}" ]]; then
    printf 'error: ONNX model does not exist: %s\n' "${model_path}" >&2
    exit 2
fi

if [[ "${model_path}" == "${engine_path}" ]]; then
    printf 'error: ENGINE_PATH must differ from the ONNX model path\n' >&2
    exit 2
fi

engine_stem="${engine_path%.engine}"
build_log="${engine_stem}.build.log"
benchmark_log="${engine_stem}.benchmark.log"
times_json="${engine_stem}.times.json"
shape="${input_name}:${batch_size}x3x${image_size}x${image_size}"

mkdir -p "$(dirname -- "${engine_path}")"

if [[ "${rebuild}" == "1" || ! -s "${engine_path}" ]]; then
    printf 'Building FP16 TensorRT engine: %s\n' "${engine_path}"
    "${trtexec}" \
        --onnx="${model_path}" \
        --fp16 \
        --memPoolSize="workspace:${workspace_mib}" \
        --minShapes="${shape}" \
        --optShapes="${shape}" \
        --maxShapes="${shape}" \
        --saveEngine="${engine_path}" \
        --buildOnly \
        2>&1 | tee "${build_log}"
else
    printf 'Reusing existing TensorRT engine: %s\n' "${engine_path}"
fi

if [[ ! -s "${engine_path}" ]]; then
    printf 'error: TensorRT engine was not created: %s\n' "${engine_path}" >&2
    exit 1
fi

printf 'Benchmarking TensorRT engine with shape %s\n' "${shape}"
"${trtexec}" \
    --loadEngine="${engine_path}" \
    --shapes="${shape}" \
    --warmUp="${warmup_ms}" \
    --duration="${duration_s}" \
    --useCudaGraph \
    --exportTimes="${times_json}" \
    2>&1 | tee "${benchmark_log}"

printf 'Engine: %s\n' "${engine_path}"
printf 'Build log: %s\n' "${build_log}"
printf 'Benchmark log: %s\n' "${benchmark_log}"
printf 'Timing JSON: %s\n' "${times_json}"
