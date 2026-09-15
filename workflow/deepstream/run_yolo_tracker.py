#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GLib", "2.0")
from gi.repository import GLib, Gst

import pyds


DEEPSTREAM_ROOT = Path("/opt/nvidia/deepstream/deepstream-6.2")
TRACKER_LIBRARY = DEEPSTREAM_ROOT / "lib/libnvds_nvmultiobjecttracker.so"
TRACKER_CONFIG = (
    DEEPSTREAM_ROOT
    / "samples/configs/deepstream-app/config_tracker_NvSORT.yml"
)


def make_element(factory, name):
    element = Gst.ElementFactory.make(factory, name)
    if element is None:
        raise RuntimeError(f"Unable to create GStreamer element: {factory}")
    return element


def link_many(*elements):
    for source, target in zip(elements, elements[1:]):
        if not source.link(target):
            raise RuntimeError(
                f"Unable to link {source.get_name()} -> {target.get_name()}"
            )


def video_size(path):
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=s=x:p=0",
        str(path),
    ]
    width, height = subprocess.check_output(command, text=True).strip().split("x")
    return int(width), int(height)


def render_infer_config(template, destination, engine, labels, parser):
    text = template.read_text(encoding="utf-8")
    replacements = {
        "@ENGINE_PATH@": str(engine),
        "@LABEL_PATH@": str(labels),
        "@PARSER_PATH@": str(parser),
    }
    for key, value in replacements.items():
        text = text.replace(key, value)
    destination.write_text(text, encoding="utf-8")


class MetadataWriter:
    def __init__(
        self,
        jsonl_path,
        action_jsonl_path=None,
        action_names=None,
        action_stride=30,
        fps=1.0,
    ):
        self.path = jsonl_path
        self.output = jsonl_path.open("w", encoding="utf-8")
        self.action_output = (
            action_jsonl_path.open("w", encoding="utf-8")
            if action_jsonl_path
            else None
        )
        self.action_names = action_names or {}
        self.action_stride = action_stride
        self.fps = fps
        self.latest_actions = {}
        self.action_events = []
        self.frames = 0
        self.detections = 0
        self.tracks = defaultdict(
            lambda: {"first_frame": None, "last_frame": None, "observations": 0}
        )

    def close(self):
        if not self.output.closed:
            self.output.close()
        if self.action_output and not self.action_output.closed:
            self.action_output.close()

    def read_action(self, object_meta, frame_number, track_id):
        classifier_list = object_meta.classifier_meta_list
        newest = None
        while classifier_list is not None:
            try:
                classifier_meta = pyds.NvDsClassifierMeta.cast(
                    classifier_list.data
                )
            except StopIteration:
                break
            label_list = classifier_meta.label_info_list
            while label_list is not None:
                try:
                    label_info = pyds.NvDsLabelInfo.cast(label_list.data)
                except StopIteration:
                    break
                act_id = int(label_info.result_class_id)
                label = self.action_names.get(act_id, label_info.result_label)
                newest = {
                    "track_id": track_id,
                    "end_frame": frame_number,
                    "covered_frames": self.action_stride,
                    "duration_seconds": round(
                        self.action_stride / self.fps, 6
                    ),
                    "act_id": act_id,
                    "action": label,
                    "confidence": round(float(label_info.result_prob), 6),
                }
                try:
                    label_list = label_list.next
                except StopIteration:
                    break
            try:
                classifier_list = classifier_list.next
            except StopIteration:
                break

        if newest is not None:
            self.latest_actions[track_id] = newest
            self.action_events.append(newest)
            if self.action_output:
                self.action_output.write(
                    json.dumps(newest, ensure_ascii=False) + "\n"
                )
        return self.latest_actions.get(track_id)

    def probe(self, _pad, info):
        gst_buffer = info.get_buffer()
        if gst_buffer is None:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        frame_list = batch_meta.frame_meta_list
        while frame_list is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(frame_list.data)
            except StopIteration:
                break

            frame_number = int(frame_meta.frame_num)
            objects = []
            object_list = frame_meta.obj_meta_list
            while object_list is not None:
                try:
                    object_meta = pyds.NvDsObjectMeta.cast(object_list.data)
                except StopIteration:
                    break

                track_id = int(object_meta.object_id)
                rect = object_meta.rect_params
                confidence = float(object_meta.confidence)
                action = self.read_action(
                    object_meta, frame_number, track_id
                )
                object_record = {
                        "track_id": track_id,
                        "class_id": int(object_meta.class_id),
                        "label": object_meta.obj_label,
                        "confidence": round(confidence, 6),
                        "bbox": [
                            round(float(rect.left), 3),
                            round(float(rect.top), 3),
                            round(float(rect.width), 3),
                            round(float(rect.height), 3),
                        ],
                    }
                if action:
                    object_record["action"] = {
                        key: action[key]
                        for key in ("act_id", "action", "confidence")
                    }
                objects.append(object_record)
                track = self.tracks[track_id]
                if track["first_frame"] is None:
                    track["first_frame"] = frame_number
                track["last_frame"] = frame_number
                track["observations"] += 1

                object_meta.text_params.display_text = (
                    f"worker #{track_id} {action['action']} "
                    f"{action['confidence']:.2f}"
                    if action
                    else f"worker #{track_id} {confidence:.2f}"
                )
                object_meta.rect_params.border_width = 2
                object_meta.rect_params.border_color.set(1.0, 0.0, 0.0, 1.0)

                try:
                    object_list = object_list.next
                except StopIteration:
                    break

            record = {
                "frame_id": frame_number,
                "pts_seconds": round(float(frame_meta.buf_pts) / Gst.SECOND, 6),
                "source_id": int(frame_meta.source_id),
                "objects": objects,
            }
            self.output.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.frames += 1
            self.detections += len(objects)

            try:
                frame_list = frame_list.next
            except StopIteration:
                break
        return Gst.PadProbeReturn.OK

    def summary(self, elapsed_seconds, input_path, output_path):
        tracks = {
            str(track_id): values
            for track_id, values in sorted(self.tracks.items())
        }
        summary = {
            "input": str(input_path),
            "output": str(output_path),
            "frames": self.frames,
            "detections": self.detections,
            "unique_tracks": len(tracks),
            "elapsed_seconds": round(elapsed_seconds, 6),
            "end_to_end_fps": round(
                self.frames / elapsed_seconds if elapsed_seconds else 0.0, 3
            ),
            "tracks": tracks,
        }
        if self.action_events:
            work = defaultdict(lambda: defaultdict(float))
            for event in self.action_events:
                work[str(event["track_id"])][event["action"]] += event[
                    "duration_seconds"
                ]
            summary["action_side_path"] = {
                "backend": "deepstream-nvdspreprocess-nvinfer",
                "events": len(self.action_events),
                "stride": self.action_stride,
                "work_seconds_by_track": {
                    track_id: {
                        action: round(seconds, 6)
                        for action, seconds in sorted(actions.items())
                    }
                    for track_id, actions in sorted(work.items())
                },
            }
        return summary


def build_pipeline(args, infer_config, writer, mux_width, mux_height):
    pipeline = Gst.Pipeline.new("yolo-tracker-pipeline")
    if pipeline is None:
        raise RuntimeError("Unable to create GStreamer pipeline")

    source = make_element("filesrc", "source")
    demuxer = make_element("qtdemux", "demuxer")
    input_parser = make_element("h264parse", "input-parser")
    decoder = make_element("nvv4l2decoder", "decoder")
    streammux = make_element("nvstreammux", "streammux")
    detector = make_element("nvinfer", "detector")
    tracker = make_element("nvtracker", "tracker")
    action_preprocess = None
    action_infer = None
    if getattr(args, "action_preprocess_config", None):
        action_preprocess = make_element("nvdspreprocess", "action-preprocess")
        action_infer = make_element("nvinfer", "action-infer")
    rgba_converter = make_element("nvvideoconvert", "rgba-converter")
    rgba_filter = make_element("capsfilter", "rgba-filter")
    osd = make_element("nvdsosd", "osd")
    nv12_converter = make_element("nvvideoconvert", "nv12-converter")
    nv12_filter = make_element("capsfilter", "nv12-filter")
    encoder = make_element("nvv4l2h264enc", "encoder")
    output_parser = make_element("h264parse", "output-parser")
    muxer = make_element("qtmux", "muxer")
    sink = make_element("filesink", "sink")

    for element in (
        source,
        demuxer,
        input_parser,
        decoder,
        streammux,
        detector,
        tracker,
        action_preprocess,
        action_infer,
        rgba_converter,
        rgba_filter,
        osd,
        nv12_converter,
        nv12_filter,
        encoder,
        output_parser,
        muxer,
        sink,
    ):
        if element is not None:
            pipeline.add(element)

    source.set_property("location", str(args.input))
    streammux.set_property("batch-size", 1)
    streammux.set_property("width", mux_width)
    streammux.set_property("height", mux_height)
    streammux.set_property("live-source", False)
    streammux.set_property("batched-push-timeout", 40000)
    detector.set_property("config-file-path", str(infer_config))

    tracker.set_property("tracker-width", 640)
    tracker.set_property("tracker-height", 384)
    tracker.set_property("ll-lib-file", str(TRACKER_LIBRARY))
    tracker.set_property("ll-config-file", str(TRACKER_CONFIG))
    tracker.set_property("gpu-id", 0)
    tracker.set_property("enable-batch-process", True)
    tracker.set_property("display-tracking-id", True)

    if action_preprocess is not None:
        action_preprocess.set_property(
            "config-file", str(args.action_preprocess_config)
        )
        action_infer.set_property(
            "config-file-path", str(args.action_infer_config)
        )

    rgba_filter.set_property(
        "caps", Gst.Caps.from_string("video/x-raw(memory:NVMM),format=RGBA")
    )
    osd.set_property("process-mode", 1)
    osd.set_property("display-text", True)
    nv12_filter.set_property(
        "caps", Gst.Caps.from_string("video/x-raw(memory:NVMM),format=NV12")
    )
    encoder.set_property("bitrate", args.bitrate)
    encoder.set_property("iframeinterval", 10)
    encoder.set_property("insert-sps-pps", True)
    muxer.set_property("faststart", True)
    sink.set_property("location", str(args.output))
    sink.set_property("sync", False)

    link_many(source, demuxer)
    link_many(input_parser, decoder)

    def on_demux_pad_added(_demuxer, new_pad):
        caps = new_pad.get_current_caps() or new_pad.query_caps(None)
        if not caps or not caps.to_string().startswith("video/"):
            return
        parser_sink = input_parser.get_static_pad("sink")
        if not parser_sink.is_linked():
            result = new_pad.link(parser_sink)
            if result != Gst.PadLinkReturn.OK:
                raise RuntimeError(f"Unable to link demuxer pad: {result}")

    demuxer.connect("pad-added", on_demux_pad_added)
    decoder_src = decoder.get_static_pad("src")
    streammux_sink = streammux.get_request_pad("sink_0")
    if decoder_src.link(streammux_sink) != Gst.PadLinkReturn.OK:
        raise RuntimeError("Unable to link decoder to nvstreammux")

    processing_chain = [streammux, detector, tracker]
    if action_preprocess is not None:
        processing_chain.extend([action_preprocess, action_infer])
    processing_chain.extend([
        rgba_converter,
        rgba_filter,
        osd,
        nv12_converter,
        nv12_filter,
        encoder,
        output_parser,
        muxer,
        sink,
    ])
    link_many(*processing_chain)

    roi_expander = getattr(args, "roi_expander", None)
    if roi_expander is not None:
        tracker.get_static_pad("src").add_probe(
            Gst.PadProbeType.BUFFER, roi_expander.expand
        )
        action_preprocess.get_static_pad("src").add_probe(
            Gst.PadProbeType.BUFFER, roi_expander.restore
        )

    osd_sink_pad = osd.get_static_pad("sink")
    osd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, writer.probe)
    return pipeline


def parse_args():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[1]
    parser = argparse.ArgumentParser(
        description="DeepStream YOLOv5 + NvSORT zero-copy video pipeline"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=project_root / "benchmark/e2e/clip1_30s.mp4",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "benchmark/deepstream/yolo_tracker_output.mp4",
    )
    parser.add_argument(
        "--engine",
        type=Path,
        default=(
            project_root
            / "yolo/edge_pt/onnx/worker.640.yolov5n6.fp16.engine"
        ),
    )
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--bitrate", type=int, default=10_000_000)
    return parser.parse_args()


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    args.input = args.input.resolve()
    args.output = args.output.resolve()
    args.engine = args.engine.resolve()
    args.metadata = (
        args.metadata.resolve()
        if args.metadata
        else args.output.with_suffix(".tracks.jsonl")
    )
    args.summary = (
        args.summary.resolve()
        if args.summary
        else args.output.with_suffix(".summary.json")
    )

    required = [
        args.input,
        args.engine,
        script_dir / "libnvdsinfer_custom_yolov5.so",
        script_dir / "labels.txt",
        script_dir / "config_infer_yolov5.txt.in",
        TRACKER_LIBRARY,
        TRACKER_CONFIG,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required files: " + ", ".join(missing))
    for path in (args.output, args.metadata, args.summary):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    input_width, input_height = video_size(args.input)
    mux_width = (input_width + 7) // 8 * 8
    mux_height = (input_height + 1) // 2 * 2
    infer_config = args.summary.with_suffix(".infer.txt")
    render_infer_config(
        script_dir / "config_infer_yolov5.txt.in",
        infer_config,
        args.engine,
        script_dir / "labels.txt",
        script_dir / "libnvdsinfer_custom_yolov5.so",
    )

    Gst.init(None)
    writer = MetadataWriter(args.metadata)
    pipeline = build_pipeline(args, infer_config, writer, mux_width, mux_height)
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
        pipeline.set_state(Gst.State.NULL)
        writer.close()
    elapsed = time.perf_counter() - started
    if error_message:
        raise RuntimeError(error_message)

    summary = writer.summary(elapsed, args.input, args.output)
    summary["input_width"] = input_width
    summary["input_height"] = input_height
    summary["output_width"] = mux_width
    summary["output_height"] = mux_height
    summary["metadata"] = str(args.metadata)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print("TRACKER_RESULT=" + json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
