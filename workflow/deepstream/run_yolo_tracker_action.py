#!/usr/bin/env python3
"""DeepStream main path with an asynchronous 3D action-recognition side path."""

import argparse
import json
import queue
import subprocess
import sys
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import gi
import numpy as np

gi.require_version("Gst", "1.0")
gi.require_version("GLib", "2.0")
from gi.repository import GLib, Gst

import pyds

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
WORKFLOW_ROOT = PROJECT_ROOT / "workflow"
if str(WORKFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKFLOW_ROOT))

from itfact.action_recognition import ActionRecognition
from itfact.tensorrt_action import TensorRTActionRecognition
from deepstream.run_yolo_tracker import (
    MetadataWriter,
    build_pipeline,
    render_infer_config,
    video_size,
)


ACTION_NAMES = {
    0: "clean",
    1: "concrete",
    2: "formwork",
    3: "prepare",
    4: "rebar",
    5: "rest/talk",
    6: "scaffold",
    7: "transport",
    8: "walk",
}
UNTRACKED_OBJECT_ID = (1 << 64) - 1


@dataclass
class ActionJob:
    track_id: int
    start_frame: int
    end_frame: int
    frames: list
    covered_frames: int


def video_fps(path):
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    numerator, denominator = (
        subprocess.check_output(command, text=True).strip().split("/")
    )
    return float(numerator) / float(denominator)


class AsyncActionSidePath:
    def __init__(
        self,
        model_path,
        backend,
        output_path,
        fps,
        window_length=30,
        sample_count=16,
        stride=30,
        batch_size=4,
        queue_size=16,
        roi_scale=2.0,
        input_size=240,
    ):
        self.model_path = model_path
        self.backend = backend
        self.output_path = output_path
        self.fps = fps
        self.window_length = window_length
        self.sample_count = sample_count
        self.stride = stride
        self.batch_size = batch_size
        self.roi_scale = roi_scale
        self.input_size = input_size
        self.jobs = queue.Queue(maxsize=queue_size)
        self.buffers = defaultdict(lambda: deque(maxlen=window_length))
        self.last_submitted = {}
        self.latest = {}
        self.events = []
        self.dropped_jobs = 0
        self.submitted_jobs = 0
        self.completed_jobs = 0
        self.inference_seconds = 0.0
        self._lock = threading.Lock()
        self._error = None
        self._thread = threading.Thread(
            target=self._worker,
            name="action-recognition",
            daemon=True,
        )
        self._thread.start()

    def _worker(self):
        try:
            if self.backend == "tensorrt":
                model = TensorRTActionRecognition(self.model_path, device=0)
            else:
                model = ActionRecognition(
                    self.model_path,
                    device=0,
                    input_size=self.input_size,
                )
            while True:
                first = self.jobs.get()
                if first is None:
                    self.jobs.task_done()
                    break
                batch = [first]
                while len(batch) < self.batch_size:
                    try:
                        item = self.jobs.get_nowait()
                    except queue.Empty:
                        break
                    if item is None:
                        self.jobs.task_done()
                        break
                    batch.append(item)

                started = time.perf_counter()
                results = model.detect_batch([job.frames for job in batch])
                elapsed = time.perf_counter() - started
                completed_at = time.perf_counter()
                with self._lock:
                    self.inference_seconds += elapsed
                    for job, result in zip(batch, results):
                        probabilities = result["logits"].detach().float().cpu()
                        confidence = float(probabilities[result["act_id"]].item())
                        event = {
                            "track_id": job.track_id,
                            "start_frame": job.start_frame,
                            "end_frame": job.end_frame,
                            "covered_frames": job.covered_frames,
                            "duration_seconds": round(
                                job.covered_frames / self.fps, 6
                            ),
                            "act_id": result["act_id"],
                            "action": ACTION_NAMES.get(
                                result["act_id"], str(result["act_id"])
                            ),
                            "confidence": round(confidence, 6),
                            "completed_at": completed_at,
                        }
                        self.events.append(event)
                        self.latest[job.track_id] = event
                        self.completed_jobs += 1
                for _ in batch:
                    self.jobs.task_done()
        except Exception as exc:
            self._error = exc
            while True:
                try:
                    self.jobs.get_nowait()
                    self.jobs.task_done()
                except queue.Empty:
                    break

    def _crop(self, surface, rect):
        height, width = surface.shape[:2]
        center_x = float(rect.left) + float(rect.width) / 2
        center_y = float(rect.top) + float(rect.height) / 2
        side = max(float(rect.width), float(rect.height)) * self.roi_scale
        side = max(2, min(int(round(side)), width, height))
        left = int(round(center_x - side / 2))
        top = int(round(center_y - side / 2))
        left = min(max(0, left), width - side)
        top = min(max(0, top), height - side)
        crop = surface[top : top + side, left : left + side, :3]
        crop = cv2.resize(
            crop,
            (self.input_size, self.input_size),
            interpolation=cv2.INTER_LINEAR,
        )
        # NvBufSurface RGBA is exposed in RGBA channel order.
        return np.ascontiguousarray(crop)

    def _submit(self, track_id, samples, end_frame):
        covered_frames = (
            self.window_length
            if track_id not in self.last_submitted
            else min(self.stride, self.window_length)
        )
        indices = np.linspace(
            0,
            len(samples) - 1,
            num=self.sample_count,
        ).round().astype(int)
        job = ActionJob(
            track_id=track_id,
            start_frame=int(samples[0][0]),
            end_frame=end_frame,
            frames=[samples[index][1] for index in indices],
            covered_frames=covered_frames,
        )
        try:
            self.jobs.put_nowait(job)
            self.submitted_jobs += 1
        except queue.Full:
            self.dropped_jobs += 1
        finally:
            # A full queue must not cause a retry on every following frame.
            self.last_submitted[track_id] = end_frame

    def probe(self, _pad, info):
        gst_buffer = info.get_buffer()
        if gst_buffer is None or self._error is not None:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        frame_list = batch_meta.frame_meta_list
        while frame_list is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(frame_list.data)
            except StopIteration:
                break

            objects = []
            object_list = frame_meta.obj_meta_list
            while object_list is not None:
                try:
                    object_meta = pyds.NvDsObjectMeta.cast(object_list.data)
                except StopIteration:
                    break
                track_id = int(object_meta.object_id)
                if track_id != UNTRACKED_OBJECT_ID:
                    objects.append((track_id, object_meta))
                    with self._lock:
                        latest = self.latest.get(track_id)
                    if latest:
                        object_meta.text_params.display_text = (
                            f"worker #{track_id} {latest['action']} "
                            f"{latest['confidence']:.2f}"
                        )
                try:
                    object_list = object_list.next
                except StopIteration:
                    break

            if objects:
                surface = np.asarray(
                    pyds.get_nvds_buf_surface(
                        hash(gst_buffer), int(frame_meta.batch_id)
                    )
                )
                frame_number = int(frame_meta.frame_num)
                for track_id, object_meta in objects:
                    samples = self.buffers[track_id]
                    samples.append(
                        (frame_number, self._crop(surface, object_meta.rect_params))
                    )
                    last = self.last_submitted.get(track_id)
                    if len(samples) == self.window_length and (
                        last is None or frame_number - last >= self.stride
                    ):
                        self._submit(track_id, list(samples), frame_number)

            try:
                frame_list = frame_list.next
            except StopIteration:
                break
        return Gst.PadProbeReturn.OK

    def finish(self):
        self.jobs.join()
        if self._thread.is_alive():
            self.jobs.put(None)
            self.jobs.join()
            self._thread.join()
        if self._error is not None:
            raise RuntimeError(f"Action worker failed: {self._error}")

        ordered = sorted(
            self.events,
            key=lambda event: (event["end_frame"], event["track_id"]),
        )
        with self.output_path.open("w", encoding="utf-8") as output:
            for event in ordered:
                event = dict(event)
                event.pop("completed_at", None)
                output.write(json.dumps(event, ensure_ascii=False) + "\n")
        return ordered

    def summary(self, events):
        work = defaultdict(lambda: defaultdict(float))
        for event in events:
            work[str(event["track_id"])][event["action"]] += event[
                "duration_seconds"
            ]
        return {
            "model": str(self.model_path),
            "backend": self.backend,
            "window_length": self.window_length,
            "sample_count": self.sample_count,
            "input_size": self.input_size,
            "stride": self.stride,
            "batch_size": self.batch_size,
            "submitted_jobs": self.submitted_jobs,
            "completed_jobs": self.completed_jobs,
            "dropped_jobs": self.dropped_jobs,
            "inference_seconds": round(self.inference_seconds, 6),
            "events": str(self.output_path),
            "work_seconds_by_track": {
                track_id: {
                    action: round(seconds, 6)
                    for action, seconds in sorted(actions.items())
                }
                for track_id, actions in sorted(work.items())
            },
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description="DeepStream YOLO + NvSORT + asynchronous 3D actions"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "benchmark/e2e/clip1_30s.mp4",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "benchmark/deepstream/yolo_tracker_action.mp4",
    )
    parser.add_argument(
        "--engine",
        type=Path,
        default=(
            PROJECT_ROOT
            / "yolo/edge_pt/onnx/worker.640.yolov5n6.fp16.engine"
        ),
    )
    parser.add_argument(
        "--action-model",
        type=Path,
        default=WORKFLOW_ROOT / "resource/jit.end2end.pt",
    )
    parser.add_argument(
        "--action-engine",
        type=Path,
        default=(
            WORKFLOW_ROOT
            / "resource/action_resnet3d_t30_s200.fp16.engine"
        ),
    )
    parser.add_argument(
        "--action-backend",
        choices=("auto", "tensorrt", "torchscript"),
        default="auto",
    )
    parser.add_argument("--action-window", type=int, default=30)
    parser.add_argument("--action-samples", type=int, default=30)
    parser.add_argument("--action-input-size", type=int, default=200)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--actions", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--action-stride", type=int, default=30)
    parser.add_argument("--action-batch-size", type=int, default=4)
    parser.add_argument("--action-queue-size", type=int, default=16)
    parser.add_argument("--bitrate", type=int, default=10_000_000)
    return parser.parse_args()


def main():
    args = parse_args()
    args.input = args.input.resolve()
    args.output = args.output.resolve()
    args.engine = args.engine.resolve()
    args.action_model = args.action_model.resolve()
    args.action_engine = args.action_engine.resolve()
    args.metadata = (
        args.metadata.resolve()
        if args.metadata
        else args.output.with_suffix(".tracks.jsonl")
    )
    args.actions = (
        args.actions.resolve()
        if args.actions
        else args.output.with_suffix(".actions.jsonl")
    )
    args.summary = (
        args.summary.resolve()
        if args.summary
        else args.output.with_suffix(".summary.json")
    )
    if min(
        args.action_window,
        args.action_samples,
        args.action_input_size,
        args.action_stride,
        args.action_batch_size,
    ) <= 0:
        raise ValueError("Action dimensions, stride, and batch size must be positive")
    if args.action_samples > args.action_window:
        raise ValueError("Action samples cannot exceed the source window")

    action_backend = args.action_backend
    if action_backend == "auto":
        action_backend = (
            "tensorrt" if args.action_engine.exists() else "torchscript"
        )
    action_path_value = (
        args.action_engine if action_backend == "tensorrt" else args.action_model
    )

    required = [
        args.input,
        args.engine,
        action_path_value,
        SCRIPT_DIR / "libnvdsinfer_custom_yolov5.so",
        SCRIPT_DIR / "labels.txt",
        SCRIPT_DIR / "config_infer_yolov5.txt.in",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required files: " + ", ".join(missing))
    for path in (args.output, args.metadata, args.actions, args.summary):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    input_width, input_height = video_size(args.input)
    fps = video_fps(args.input)
    mux_width = (input_width + 7) // 8 * 8
    mux_height = (input_height + 1) // 2 * 2
    infer_config = args.summary.with_suffix(".infer.txt")
    render_infer_config(
        SCRIPT_DIR / "config_infer_yolov5.txt.in",
        infer_config,
        args.engine,
        SCRIPT_DIR / "labels.txt",
        SCRIPT_DIR / "libnvdsinfer_custom_yolov5.so",
    )

    Gst.init(None)
    metadata_writer = MetadataWriter(args.metadata)
    action_path = AsyncActionSidePath(
        action_path_value,
        action_backend,
        args.actions,
        fps=fps,
        window_length=args.action_window,
        sample_count=args.action_samples,
        stride=args.action_stride,
        batch_size=args.action_batch_size,
        queue_size=args.action_queue_size,
        input_size=args.action_input_size,
    )
    pipeline = build_pipeline(
        args, infer_config, metadata_writer, mux_width, mux_height
    )
    pipeline.get_by_name("osd").get_static_pad("sink").add_probe(
        Gst.PadProbeType.BUFFER, action_path.probe
    )
    loop = GLib.MainLoop()
    error_message = None

    def on_bus_message(_bus, message):
        nonlocal error_message
        if message.type == Gst.MessageType.EOS:
            loop.quit()
        elif message.type == Gst.MessageType.ERROR:
            error, debug = message.parse_error()
            error_message = f"{error}: {debug}"
            loop.quit()
        return True

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_bus_message)

    started = time.perf_counter()
    if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
        raise RuntimeError("Unable to set the pipeline to PLAYING")
    try:
        loop.run()
    finally:
        video_elapsed = time.perf_counter() - started
        pipeline.set_state(Gst.State.NULL)
        metadata_writer.close()
    if error_message:
        raise RuntimeError(error_message)

    events = action_path.finish()
    total_elapsed = time.perf_counter() - started
    summary = metadata_writer.summary(
        video_elapsed, args.input, args.output
    )
    summary.update(
        {
            "input_width": input_width,
            "input_height": input_height,
            "output_width": mux_width,
            "output_height": mux_height,
            "metadata": str(args.metadata),
            "video_pipeline_seconds": round(video_elapsed, 6),
            "video_pipeline_fps": round(
                metadata_writer.frames / video_elapsed, 3
            ),
            "total_with_action_drain_seconds": round(total_elapsed, 6),
            "total_with_action_drain_fps": round(
                metadata_writer.frames / total_elapsed, 3
            ),
            "action_side_path": action_path.summary(events),
        }
    )
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print("ACTION_RESULT=" + json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
