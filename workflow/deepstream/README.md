# DeepStream YOLOv5 zero-copy benchmark

This pipeline keeps decoded video surfaces in Jetson NVMM through TensorRT
inference, OSD, and hardware H.264 encoding. Only detection metadata is copied
to the CPU.

Requirements: JetPack 5.1.1, DeepStream 6.2, TensorRT 8.5, and the fixed-shape
FP16 engine generated for `images:1x3x640x640`.

Run from the repository root:

```bash
bash workflow/deepstream/run_yolo_zero_copy.sh [INPUT_MP4] [OUTPUT_MP4] [ENGINE]
```

The script refuses to overwrite an existing output. Per-frame detection
metadata, generated configs, and the DeepStream log are retained under
`benchmark/deepstream/run.*` for inspection.

`nvstreammux` aligns output width to a multiple of eight for Jetson NVMM. For
example, a 2490-pixel-wide input produces a 2496-pixel-wide output surface.

The validated Orin NX results are recorded in [BENCHMARK.md](BENCHMARK.md) and
[benchmark_results.json](benchmark_results.json).

## Tracking metadata

The NvSORT pipeline keeps video frames in NVMM and copies only object metadata
to Python. It writes one JSON object per frame and a track summary:

```bash
bash workflow/deepstream/run_yolo_tracker.sh \
  --input benchmark/e2e/clip1_30s.mp4 \
  --output benchmark/deepstream/yolo_tracker_output.mp4
```

The default metadata outputs are `yolo_tracker_output.tracks.jsonl` and
`yolo_tracker_output.summary.json`.

## Asynchronous action recognition

The action pipeline adds a non-blocking side path after NvSORT. The video path
continues through NVMM, OSD, and the hardware encoder; only tracked worker ROIs
are resized and queued for the 3D ResNet TensorRT model. Action jobs are batched
and processed on a background thread.

Build the action engine once on the target Jetson (the engine is tied to its
TensorRT/GPU environment):

```bash
bash workflow/deepstream/build_action_engine.sh
```

```bash
bash workflow/deepstream/run_yolo_tracker_action.sh \
  --input benchmark/e2e/clip1_30s.mp4 \
  --output benchmark/deepstream/yolo_tracker_action.mp4
```

The command produces four result artifacts:

- annotated MP4 video;
- per-frame track metadata in `*.tracks.jsonl`;
- per-window action results in `*.actions.jsonl`;
- a `*.summary.json` file containing both video-path throughput and total
  latency after draining pending action work, plus per-worker action seconds.

`--action-stride` controls how often a 30-frame clip is classified. The default
is 30 frames, which avoids overlapping accounting windows. The action queue is
deliberately non-blocking: if action inference cannot keep up, dropped jobs are
reported in the summary instead of stalling the real-time video path.

The validated edge profile keeps all 30 temporal frames and uses 200x200 worker
ROIs. A more aggressive 16-frame/120x120 profile was rejected because it only
matched 3 of 6 reference predictions. The 30-frame/200x200 profile matched all
6 reference windows; this agreement check is a regression test, not a substitute
for accuracy evaluation on a labeled validation set:

```bash
PYTHONPATH=/usr/lib/python3.8/dist-packages \
  /home/p/mambaforge/envs/python38/bin/python \
  workflow/itfact/validate_action_tensorrt.py
```

## Native tracked-sequence path

For the fully GPU-side temporal path, use the native sequence preprocessor. It
keeps the video surfaces in NVMM and lets `nvdspreprocess` accumulate the 30
frames before `nvinfer` runs the 3D classifier. The custom library keys each
sequence by `(source_id, NvSORT object_id)`, so a moving worker keeps one stable
temporal window even though its ROI coordinates change every frame. Untracked
objects use the original ROI key as a safe fallback.

Build and run this variant on the Jetson (DeepStream 6.2):

```bash
bash workflow/deepstream/run_yolo_tracker_action_native.sh \
  --input benchmark/e2e/clip1_30s.mp4 \
  --output benchmark/deepstream/yolo_tracker_action_native.mp4
```

`build_track_sequence_preprocess.sh` copies and patches NVIDIA's
`custom_sequence_preprocess` sample into
`libnvds_custom_sequence_track_id.so`. Python only reads NvDsClassifierMeta for
action labels/confidences and writes the JSONL/summary artifacts; it does not
copy ROI pixels to the CPU. The script refuses to overwrite existing generated
artifacts or an existing custom library.
