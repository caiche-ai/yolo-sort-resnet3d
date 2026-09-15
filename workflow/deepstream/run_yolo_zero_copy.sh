#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/../.." && pwd)"

input_path="${1:-${project_root}/benchmark/e2e/clip1_30s.mp4}"
output_path="${2:-${project_root}/benchmark/deepstream/yolo_zero_copy_output.mp4}"
engine_path="${3:-${project_root}/yolo/edge_pt/onnx/worker.640.yolov5n6.fp16.engine}"

for command_name in deepstream-app gst-inspect-1.0 ffprobe; do
    if ! command -v "${command_name}" >/dev/null 2>&1; then
        printf 'error: required command is unavailable: %s\n' "${command_name}" >&2
        exit 1
    fi
done
for plugin_name in nvinfer nvstreammux nvdsosd nvvideoconvert nvv4l2h264enc; do
    if ! gst-inspect-1.0 "${plugin_name}" >/dev/null 2>&1; then
        printf 'error: required GStreamer plugin is unavailable: %s\n' "${plugin_name}" >&2
        exit 1
    fi
done

input_path="$(realpath "${input_path}")"
engine_path="$(realpath "${engine_path}")"
output_path="$(realpath -m "${output_path}")"
if [[ -e "${output_path}" ]]; then
    printf 'error: output already exists: %s\n' "${output_path}" >&2
    exit 1
fi

mkdir -p "$(dirname "${output_path}")"
mkdir -p "${project_root}/benchmark/deepstream"
run_dir="$(mktemp -d "${project_root}/benchmark/deepstream/run.XXXXXX")"
metadata_dir="${run_dir}/metadata"
mkdir -p "${metadata_dir}"

IFS=x read -r input_width input_height < <(
    ffprobe -v error -select_streams v:0 \
        -show_entries stream=width,height -of csv=s=x:p=0 "${input_path}"
)
mux_width="$(( (input_width + 7) / 8 * 8 ))"
mux_height="$(( (input_height + 1) / 2 * 2 ))"

make -C "${script_dir}"
parser_path="$(realpath "${script_dir}/libnvdsinfer_custom_yolov5.so")"
label_path="$(realpath "${script_dir}/labels.txt")"
infer_config="${run_dir}/config_infer_yolov5.txt"
app_config="${run_dir}/config_app_yolov5.txt"
run_log="${run_dir}/deepstream.log"

sed \
    -e "s|@ENGINE_PATH@|${engine_path}|g" \
    -e "s|@LABEL_PATH@|${label_path}|g" \
    -e "s|@PARSER_PATH@|${parser_path}|g" \
    "${script_dir}/config_infer_yolov5.txt.in" >"${infer_config}"

sed \
    -e "s|@INPUT_URI@|file://${input_path}|g" \
    -e "s|@OUTPUT_PATH@|${output_path}|g" \
    -e "s|@METADATA_DIR@|${metadata_dir}|g" \
    -e "s|@INFER_CONFIG@|${infer_config}|g" \
    -e "s|@MUX_WIDTH@|${mux_width}|g" \
    -e "s|@MUX_HEIGHT@|${mux_height}|g" \
    "${script_dir}/config_app_yolov5.txt.in" >"${app_config}"

started_ns="$(date +%s%N)"
set +e
deepstream-app -c "${app_config}" 2>&1 | tee "${run_log}"
deepstream_status="${PIPESTATUS[0]}"
set -e
finished_ns="$(date +%s%N)"
if [[ "${deepstream_status}" -ne 0 ]]; then
    printf 'error: deepstream-app exited with status %s; log: %s\n' \
        "${deepstream_status}" "${run_log}" >&2
    exit "${deepstream_status}"
fi

elapsed_seconds="$(awk -v start="${started_ns}" -v end="${finished_ns}" \
    'BEGIN { printf "%.6f", (end - start) / 1000000000 }')"
frame_count="$(ffprobe -v error -count_frames -select_streams v:0 \
    -show_entries stream=nb_read_frames -of default=nokey=1:noprint_wrappers=1 \
    "${output_path}")"
processing_fps="$(awk -v frames="${frame_count}" -v seconds="${elapsed_seconds}" \
    'BEGIN { printf "%.3f", frames / seconds }')"
box_count="$(find "${metadata_dir}" -type f -name '*.txt' -print0 \
    | xargs -0 -r awk 'END { print NR }' \
    | awk '{ total += $1 } END { print total + 0 }')"
box_count="${box_count:-0}"

printf 'DEEPSTREAM_RESULT frames=%s boxes=%s total_seconds=%s total_fps=%s\n' \
    "${frame_count}" "${box_count}" "${elapsed_seconds}" "${processing_fps}"
printf 'output=%s\nrun_dir=%s\n' "${output_path}" "${run_dir}"
