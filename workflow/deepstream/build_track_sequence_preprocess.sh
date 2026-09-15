#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEEPSTREAM_ROOT="${DEEPSTREAM_ROOT:-/opt/nvidia/deepstream/deepstream-6.2}"
SOURCE_DIR="${DEEPSTREAM_ROOT}/sources/apps/sample_apps/deepstream-3d-action-recognition/custom_sequence_preprocess"
OUTPUT="${SCRIPT_DIR}/libnvds_custom_sequence_track_id.so"

if [[ -e "${OUTPUT}" ]]; then
  echo "Refusing to overwrite ${OUTPUT}" >&2
  exit 1
fi
if [[ ! -d "${SOURCE_DIR}" ]]; then
  echo "DeepStream 6.2 custom sequence source is missing: ${SOURCE_DIR}" >&2
  exit 1
fi

BUILD_DIR="$(mktemp -d)"
cleanup() {
  rm -rf -- "${BUILD_DIR}"
}
trap cleanup EXIT

cp -a "${SOURCE_DIR}/." "${BUILD_DIR}/"
patch -d "${BUILD_DIR}" -p1 < "${SCRIPT_DIR}/custom_sequence_track_id.patch"
# Temporary runtime diagnostics for sequence buffering; disable after diagnosis.
if [[ "${SEQUENCE_DEBUG:-0}" == "1" ]]; then
  sed -i 's/LOG_DEBUG/LOG_INFO/g' "${BUILD_DIR}/sequence_image_process.cpp"
fi
# NVIDIA's sample Makefile uses paths relative to its original location under
# deepstream-3d-action-recognition.  The source is intentionally built in a
# temporary directory, so add absolute include paths instead of relying on
# those relative paths.
PKG_CFLAGS="$(pkg-config --cflags gstreamer-1.0)"
make -C "${BUILD_DIR}" CUDA_VER=11.4 \
  CFLAGS="-fvisibility=hidden -Wall -Werror -DPLATFORM_TEGRA \
-I${DEEPSTREAM_ROOT}/sources/includes \
-I${DEEPSTREAM_ROOT}/sources/gst-plugins/gst-nvdspreprocess/include \
-I${DEEPSTREAM_ROOT}/sources/gst-plugins/gst-nvdspreprocess/inc \
-I/usr/local/cuda-11.4/include -fPIC -std=c++14 -pthread ${PKG_CFLAGS}"
install -m 0755 \
  "${BUILD_DIR}/libnvds_custom_sequence_preprocess.so" \
  "${OUTPUT}"
echo "Built ${OUTPUT}"
