import cv2
import numpy as np
import time
import queue
import threading
import os
import logging
import shutil
import subprocess


def _gstreamer_element_available(name):
    if not shutil.which("gst-inspect-1.0"):
        return False
    return subprocess.run(
        ["gst-inspect-1.0", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def _hardware_codec_enabled():
    mode = os.environ.get("ITFVP_HW_CODEC", "auto").strip().lower()
    return mode not in {"0", "false", "off", "software"}

class RTSPInputStreamer(object):
    '''
    RTSP Stream Input
    read from RTSP stream
    '''
    def __init__(self, url):
        self.url = url
        self._cap = cv2.VideoCapture(self.url)
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self._cap.get(cv2.CAP_PROP_FPS))


    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._cap.release()

    def terminate(self):
        self._cap.release()

    def __iter__(self):
        return self

    def __next__(self):
        ret, frame = self._cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame
        else:
            raise StopIteration

class FileInputStreamer(object):
    '''
    File Stream Input
    read from video file
    '''
    def __init__(self, path, keep_rate=False):
        self.path = path[7:] if path.startswith("file://") else path
        self._cap = cv2.VideoCapture(self.path)
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self._cap.get(cv2.CAP_PROP_FPS))
        self.frame_read = 0
        self.keep_rate = keep_rate
        self._decoder_process = None

        fourcc_value = int(self._cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join(chr((fourcc_value >> 8 * i) & 0xFF) for i in range(4))
        can_use_hardware = (
            os.name != "nt"
            and _hardware_codec_enabled()
            and fourcc.lower() in {"avc1", "h264", "x264"}
            and shutil.which("gst-launch-1.0")
            and _gstreamer_element_available("nvv4l2decoder")
            and _gstreamer_element_available("nvvidconv")
        )
        if can_use_hardware:
            self._cap.release()
            self._cap = None
            command = [
                "gst-launch-1.0", "-q",
                "filesrc", f"location={self.path}", "!",
                "qtdemux", "!",
                "h264parse", "!",
                "nvv4l2decoder", "!",
                "nvvidconv", "!",
                "video/x-raw,format=BGRx", "!",
                "fdsink", "fd=1", "sync=false",
            ]
            self._decoder_process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
            logging.info("Using NVIDIA nvv4l2decoder for file input")
        else:
            logging.info("Using OpenCV software decoder for file input")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.terminate()

    def __iter__(self):
        return self
    
    def __next__(self):
        current_ts = time.time()
        if self.keep_rate:
            if self.frame_read == 0:
                self.start_ts = current_ts
            ts_gap = self.frame_read/self.fps - (current_ts - self.start_ts)
            if ts_gap > 0:
                time.sleep(ts_gap)

        if self._decoder_process is not None:
            frame_size = self.width * self.height * 4
            chunks = []
            bytes_read = 0
            while bytes_read < frame_size:
                chunk = self._decoder_process.stdout.read(frame_size - bytes_read)
                if not chunk:
                    break
                chunks.append(chunk)
                bytes_read += len(chunk)
            if bytes_read != frame_size:
                raise StopIteration
            frame = cv2.cvtColor(
                np.frombuffer(b"".join(chunks), dtype=np.uint8).reshape(
                    self.height, self.width, 4
                ),
                cv2.COLOR_BGRA2RGB,
            )
            self.frame_read += 1
            return frame

        ret, frame = self._cap.read()
        if not ret:
            raise StopIteration
        self.frame_read += 1
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def terminate(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._decoder_process is not None:
            if self._decoder_process.stdout:
                self._decoder_process.stdout.close()
            if self._decoder_process.poll() is None:
                self._decoder_process.terminate()
                try:
                    self._decoder_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._decoder_process.kill()
                    self._decoder_process.wait()
            self._decoder_process = None
        




class InputStreamer(object):
    """
    The return value of __next__ is a np.ndarray with shape (H, W, C) and dtype np.uint8 in RGB format
    """
    def __init__(self, source, keep_rate=False) -> None: 
        self.source = source
        if source.endswith(".mp4") and os.path.exists(source[7:]):
            input_streamer = FileInputStreamer(source, keep_rate)
        elif source.startswith("rtsp"):
            input_streamer = RTSPInputStreamer(source)
        else:
            raise NotImplementedError
        self.width = input_streamer.width
        self.height = input_streamer.height
        self.fps = input_streamer.fps
        # High-resolution RGB frames are large (about 10 MiB at 2490x1400).
        # Keep prefetch bounded so the shared Jetson CPU/GPU memory remains
        # available for TensorRT and the 3D action model.
        self._input_queue = queue.Queue(maxsize=8)
        self._terminate = False
        self._input_streamer = input_streamer
        self._input_thread = threading.Thread(target=self._worker_func, args=(input_streamer, self._input_queue))
        self._input_thread.daemon = True
        self._input_thread.start()

    def _worker_func(self, input_streamer, input_queue):
        """
        input_streamer: cv2.VideoCapture
        input_queue: queue.Queue
        """
        try:
            for frame in input_streamer:
                if self._terminate:
                    break
                logging.debug(f"InputQueueSize: {input_queue.qsize()}")
                input_queue.put(frame)
        finally:
            input_streamer.terminate()
            input_queue.put(None)
    
    def terminate(self):
        logging.info("terminate input streamer")
        self._terminate = True
        self._input_streamer.terminate()

    def __iter__(self):
        return self

    def __next__(self):
        while self._input_queue.empty():
            if self._terminate:
                raise StopIteration
            time.sleep(0.01)
        frame = self._input_queue.get()
        if frame is None:
            raise StopIteration
        return frame



