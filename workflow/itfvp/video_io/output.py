import os
import subprocess
import numpy as np
import cv2
import logging, queue, threading
import shutil


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

class BlankFrame(object):
    pass

class BaseOutputStreamerBase:
    def __init__(self, width, height, fps):
        self.width = width
        self.height = height
        self.fps = fps

    def write(self, frame):
        """
        frame: np.ndarray, shape=(height, width, 3), dtype=np.uint8, RGB
        """
        raise NotImplementedError

class RTSPOutputStreamer(BaseOutputStreamerBase):
    def __init__(self, width, height, fps, push_url):
        super().__init__(width, height, fps)
        self.push_url = push_url
        if os.name == "nt":
            path ='D:/ffmpeg-7.0.2-essentials_build/bin/ffmpeg.exe'
        else:
            path ='/usr/bin/ffmpeg'

        self.command = [
            path,
            '-y', '-an',
            '-f', 'rawvideo',
            '-vcodec','rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', "{}x{}".format(width, height),
            '-r', str(fps),
            '-i', '-',
            '-preset', 'veryfast',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-x264-params', 'keyint=10:bframes=0',
            '-f', 'rtsp',
            '-rtsp_transport', 'tcp',  # 使用TCP推流，linux中一定要有这行
            push_url]

        self.pipe = subprocess.Popen(self.command, shell=False, stdin=subprocess.PIPE)
        self.blank_frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    def write(self, frame):
        try:
            if isinstance(frame, BlankFrame):
                frame = self.blank_frame
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.pipe.stdin.write(frame.astype(np.uint8).tobytes())
        except BrokenPipeError:
            print("BrokenPipeError")
            self.pipe = subprocess.Popen(self.command, shell=False, stdin=subprocess.PIPE)
            self.pipe.stdin.write(frame.astype(np.uint8).tobytes())

class FileOutputStreamer(BaseOutputStreamerBase):
    def __init__(self, width, height, fps, output_path):
        super().__init__(width, height, fps)
        self.output_path = output_path
        use_hardware_encoder = (
            os.name != "nt"
            and _hardware_codec_enabled()
            and shutil.which("gst-launch-1.0")
            and _gstreamer_element_available("nvv4l2h264enc")
            and _gstreamer_element_available("nvvidconv")
        )
        self.hardware_encoder = use_hardware_encoder
        if use_hardware_encoder:
            bitrate = os.environ.get("ITFVP_H264_BITRATE", "10000000")
            self.command = [
                "gst-launch-1.0", "-q",
                "fdsrc", "fd=0", "!",
                "videoparse", "format=bgrx",
                f"width={width}", f"height={height}",
                f"framerate={fps}/1", "!",
                "nvvidconv", "!",
                "video/x-raw(memory:NVMM),format=NV12", "!",
                "nvv4l2h264enc", f"bitrate={bitrate}",
                "iframeinterval=10", "insert-sps-pps=true", "!",
                "h264parse", "!",
                "qtmux", "faststart=true", "!",
                "filesink", f"location={output_path}",
            ]
            logging.info("Using NVIDIA nvv4l2h264enc for file output")
        else:
            if os.name == "nt":
                path = 'D:/ffmpeg-7.0.2-essentials_build/bin/ffmpeg.exe'
            else:
                path = '/usr/bin/ffmpeg'
            self.command = [
                path,
                '-y', '-an',
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-pix_fmt', 'bgr24',
                '-s', "{}x{}".format(width, height),
                '-r', str(fps),
                '-i', '-',
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-preset', 'veryfast',
                '-x264-params', 'keyint=10:bframes=0',
                '-movflags', '+faststart',
                output_path,
            ]
            logging.info("Using FFmpeg libx264 software encoder for file output")

        self.pipe = subprocess.Popen(
            self.command, shell=False, stdin=subprocess.PIPE, bufsize=0
        )
        self.blank_frame = np.zeros((height, width, 3), dtype=np.uint8)

    def write(self, frame):
        if isinstance(frame, BlankFrame):
            frame = self.blank_frame
        if self.hardware_encoder:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGRA)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.pipe.stdin.write(frame.astype(np.uint8).tobytes())


class DummyOutputStreamer(BaseOutputStreamerBase):
    def __init__(self, width, height, fps):
        super().__init__(width, height, fps)
    
    def write(self, frame):
        pass


class OutputStreamer(object):
    def __init__(self, target, width, height, fps) -> None:
        self._target = target
        if target.startswith("file://"):
            target = target[7:]
            output_streamer = FileOutputStreamer(width, height, fps, target)
            self.streaming = False
        elif target.startswith("rtsp://"):
            output_streamer = RTSPOutputStreamer(width, height, fps, target)
            self.streaming = True
        elif target == "dummy":
            output_streamer = DummyOutputStreamer(width, height, fps)
            self.streaming = False
        else:
            raise NotImplementedError
        self._output_queue = queue.Queue(maxsize=30)  
        self._terminate = False
        self._sentinel_sent = False
        self._closed = False
        self._output_thread = threading.Thread(target=self._worker_func, args=(output_streamer, self._output_queue))
        self._output_thread.start()
        self._output_streamer = output_streamer  # 保存output_streamer引用，用于后续关闭

    def _worker_func(self, output_streamer, output_queue):
        while True:
            try:
                # 先获取帧，不立即判断None，确保处理完所有帧
                frame = output_queue.get(block=True, timeout=1)
                logging.debug(f"OutputQueueSize: {output_queue.qsize()}")
            except queue.Empty:
                if self._terminate:
                    break
                else:
                    continue

            # 处理帧（即使收到None，也先处理完已有帧）
            if frame is None:
                break
            output_streamer.write(frame)

    def terminate(self):
        if self._closed:
            return

        logging.info("开始终止输出流，等待剩余帧处理...")
        self._terminate = True
        if not self._sentinel_sent:
            self._output_queue.put(None)
            self._sentinel_sent = True
        # 等待队列中的帧全部编码完成。
        self._output_thread.join(timeout=30)
        if self._output_thread.is_alive():
            logging.warning("输出线程在30秒内未结束")
        # 步骤2：关闭ffmpeg的输入管道（告知ffmpeg已无数据输入，触发收尾）
        if hasattr(self._output_streamer, 'pipe') and self._output_streamer.pipe.stdin:
            try:
                self._output_streamer.pipe.stdin.close()
                logging.info("已关闭ffmpeg输入管道")
            except Exception as e:
                logging.warning(f"关闭输入管道失败: {e}")
        # 步骤3：等待ffmpeg进程完成编码（最多等待10秒，确保写入moov atom）
        if hasattr(self._output_streamer, 'pipe'):
            try:
                # wait()会阻塞直到进程结束，返回退出码
                exit_code = self._output_streamer.pipe.wait(timeout=10)
                logging.info(f"ffmpeg进程已结束，退出码: {exit_code}")
            except subprocess.TimeoutExpired:
                # 超时未结束，强制杀死进程（万不得已）
                self._output_streamer.pipe.kill()
                logging.warning("ffmpeg进程超时，已强制终止")
        self._closed = True
        logging.info("输出流终止完成")

    def write(self, frame):
        if self._terminate or self._closed:
            return
        self._output_queue.put(frame)
        if frame is None:
            self._sentinel_sent = True

