#!/usr/bin/env python3
"""Compare reduced TensorRT action predictions with the original model."""

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

from action_recognition import ActionRecognition
from tensorrt_action import TensorRTActionRecognition


def square_crop(frame, bbox, output_size, scale=2.0):
    left, top, width, height = bbox
    frame_height, frame_width = frame.shape[:2]
    center_x = left + width / 2
    center_y = top + height / 2
    side = max(width, height) * scale
    side = max(2, min(int(round(side)), frame_width, frame_height))
    crop_left = min(max(0, int(round(center_x - side / 2))), frame_width - side)
    crop_top = min(max(0, int(round(center_y - side / 2))), frame_height - side)
    crop = frame[crop_top : crop_top + side, crop_left : crop_left + side]
    return cv2.resize(crop, (output_size, output_size))


def collect_windows(video_path, metadata_path, count, window_length=30):
    records = {
        record["frame_id"]: record["objects"]
        for record in (
            json.loads(line)
            for line in metadata_path.read_text(encoding="utf-8").splitlines()
        )
    }
    buffers = defaultdict(lambda: deque(maxlen=window_length))
    windows = []
    capture = cv2.VideoCapture(str(video_path))
    frame_id = 0
    while len(windows) < count:
        ok, frame = capture.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        for obj in records.get(frame_id, []):
            track_id = int(obj["track_id"])
            samples = buffers[track_id]
            samples.append(square_crop(frame, obj["bbox"], 240))
            if len(samples) == window_length:
                windows.append(
                    {
                        "track_id": track_id,
                        "end_frame": frame_id,
                        "frames": list(samples),
                    }
                )
                samples.clear()
                if len(windows) >= count:
                    break
        frame_id += 1
    capture.release()
    return windows


def parse_args():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video", type=Path, default=root / "benchmark/e2e/clip1_30s.mp4"
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=(
            root
            / "benchmark/deepstream/clip1_30s_yolo_nvsort.tracks.jsonl"
        ),
    )
    parser.add_argument(
        "--torch-model",
        type=Path,
        default=root / "workflow/resource/jit.end2end.pt",
    )
    parser.add_argument(
        "--engine",
        type=Path,
        default=(
            root
            / "workflow/resource/action_resnet3d_t30_s200.fp16.engine"
        ),
    )
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--input-size", type=int, default=200)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    windows = collect_windows(args.video, args.metadata, args.count)
    reference = ActionRecognition(args.torch_model, device=0, input_size=240)
    reduced = TensorRTActionRecognition(args.engine, device=0)
    comparisons = []
    for window in windows:
        reference_result = reference.detect(window["frames"])
        indices = np.linspace(0, 29, num=args.samples).round().astype(int)
        reduced_frames = [
            cv2.resize(
                window["frames"][index],
                (args.input_size, args.input_size),
            )
            for index in indices
        ]
        reduced_result = reduced.detect_batch([reduced_frames])[0]
        reference_confidence = float(
            reference_result["logits"][reference_result["act_id"]].item()
        )
        reduced_confidence = float(
            reduced_result["logits"][reduced_result["act_id"]].item()
        )
        comparisons.append(
            {
                "track_id": window["track_id"],
                "end_frame": window["end_frame"],
                "reference_act_id": reference_result["act_id"],
                "reference_confidence": round(reference_confidence, 6),
                "reduced_act_id": reduced_result["act_id"],
                "reduced_confidence": round(reduced_confidence, 6),
                "top1_match": (
                    reference_result["act_id"] == reduced_result["act_id"]
                ),
            }
        )
    matches = sum(item["top1_match"] for item in comparisons)
    report = {
        "windows": len(comparisons),
        "top1_matches": matches,
        "top1_agreement": round(matches / len(comparisons), 6)
        if comparisons
        else None,
        "profile": {
            "window_length": 30,
            "samples": args.samples,
            "input_size": args.input_size,
        },
        "comparisons": comparisons,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
