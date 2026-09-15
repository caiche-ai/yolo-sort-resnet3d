# DeepStream YOLOv5 benchmark

Test date: 2026-08-27

## Environment

- Device: NVIDIA Jetson Orin NX Developer Kit
- Power mode: 15 W
- JetPack: 5.1.1 / L4T 35.3.1
- CUDA: 11.4
- TensorRT: 8.5.2
- DeepStream: 6.2.0
- Model: `worker.640.yolov5n6.fp16.engine`
- Input: H.264, 2490x1400, 10 FPS, 30.1 seconds, 301 frames

## Results

| Pipeline | Total seconds | End-to-end FPS | Notes |
|---|---:|---:|---|
| Python + ONNX Runtime CUDA | Not retained | 37.33 | Model forward only; not video-to-video |
| TensorRT `trtexec` | Not applicable | 171.38 | Random input, model forward only |
| Python + TensorRT + software codecs | 20.619 | 14.60 | Includes model initialization |
| Python + TensorRT + NVIDIA codecs | 19.794 | 15.21 | Includes model initialization |
| DeepStream zero-copy, first cold run | 11.079 | 27.17 | Includes process and model initialization |
| DeepStream zero-copy, cached rerun | 8.156 | 36.90 | Includes process and model initialization |
| DeepStream steady-state | Not applicable | 107.58 | DeepStream in-pipeline throughput after initialization |

The DeepStream output contains 301 frames and 1422 worker detections. It is an
H.264 file at 10 FPS with a duration of 30.1 seconds. Jetson `nvstreammux`
aligns the 2490-pixel input width to a 2496-pixel NVMM output surface.

## Interpreting the FPS values

The 107.58 FPS value is the steady-state throughput reported by DeepStream. It
excludes startup costs such as plugin initialization, TensorRT engine loading,
NVMM allocation, and MP4 finalization. For short offline videos, use the cold
end-to-end result (27.17 FPS) as the conservative figure. For long-running file
or RTSP processing, steady-state throughput is the relevant capacity figure.

These results cover only decode, YOLO inference, NMS, OSD, and encode. SORT and
3D ResNet action recognition are not part of this benchmark.

## Server artifacts

```text
/home/p/caic/yolo-sort-resnet3d/benchmark/deepstream/clip1_30s_deepstream_yolo_final.mp4
/home/p/caic/yolo-sort-resnet3d/benchmark/deepstream/run.0fHTNo/deepstream.log
/home/p/caic/yolo-sort-resnet3d/benchmark/deepstream/run.0fHTNo/metadata/
```
