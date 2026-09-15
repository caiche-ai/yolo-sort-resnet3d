import sys
from pathlib import Path
import cv2
import numpy as np
import torch
import queue, signal, logging
import xlsxwriter
import numpy as np
from collections import defaultdict
import time, copy
from collections import deque
from typing import List, Tuple, Union
import collections
import argparse
from itfvp.video_io import BlankFrame, InputStreamer, OutputStreamer
from itfvp.detector import YoloDetector
from itfvp.util.draw import draw_boxes
from tracker.sort import Sort
from itfact.action_recognition import ActionRecognition

# 项目路径
ROOT = Path(__file__).parent
ROOT not in sys.path and sys.path.append(ROOT)

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        #logging.FileHandler("logs/worker_action.log"),
        logging.StreamHandler()
    ]
)

# 动作类别
action_id_map = {
    0: "clean",
    1: "concrete",
    2: "formwork",
    3: "prepare",
    4: "rebar",
    5: "rest/talk",
    6: "scaffold",
    7: "transport",
    8: "walk"
}

# 在视频帧上绘制检测框和标注信息
def draw(frame, box, tid, aid, color=(0, 255, 0), thickness=2):
    x1, y1, x2, y2 = box[:4]
    tid, aid = int(tid), int(aid)
    frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    frame = cv2.putText(frame, f"worker{tid}_{action_id_map[aid]}", (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return frame

# 帧缓存类，基于双边队列deque
class FrameCache(object):
    def __init__(self, maxsize=30):
        self.maxsize = maxsize
        self.cache = deque(maxlen=maxsize)

    def append(self, frame):
        self.cache.append(frame)

    def size(self):
        return len(self.cache)

    def get(self, idx):
        return self.cache[idx]

# bbox、帧id缓存类
class BBoxBuffer(object):
    def __init__(self):
        self.buffer = []
        self.last_frame_id = -1
    
    def append(self, bbox, frame_id):
        self.buffer.append(bbox)
        self.last_frame_id = frame_id

    @property
    def size(self):
        return len(self.buffer)
    
    @property
    def first_frame_id(self):
        return self.last_frame_id - self.size + 1


def smooth(center_list):
    """平滑处理
    raw_points = [
        (10, 20),   # 点0
        (12, 22),   # 点1
        (15, 25),   # 点2
        (18, 28),   # 点3
        (20, 30),   # 点4
        (22, 32)    # 点5
    ]
    点0: 窗口[0,1) → 只有点0 → (10,20)
    点1: 窗口[0,2) → 点0,1的平均 → (11,21)
    点2: 窗口[0,3) → 点0,1,2的平均 → (12.33,22.33) → (12,22)
    点3: 窗口[1,4) → 点1,2,3的平均 → (15,25)
    点4: 窗口[2,5) → 点2,3,4的平均 → (17.67,27.67) → (18,28)
    点5: 窗口[3,6) → 点3,4,5的平均 → (20,30)
    """
    n = len(center_list)
    new_center_list = []
    for i in range(n):
        l, r = max(0, i-2), i+1
        x = round(sum(center_list[j][0] for j in range(l, r))/(r-l))
        y = round(sum(center_list[j][1] for j in range(l, r))/(r-l))
        new_center_list.append((x, y))
    return new_center_list

# 环形输出缓冲区类（环形，固定大小，满了添加新数据会覆盖原有的）
# 用于在视频处理或流处理中缓存和按序输出处理结果
class OutputBuffer(object):
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = [None] * buffer_size
        self.head_index = -1
        self.output_count = 0

    def get(self, frame_id):
        index = frame_id % self.buffer_size
        return self.buffer[index]

    def pop(self):
        if self.head_index == -1:
            return None
            
        frame = self.buffer[self.head_index]
        self.buffer[self.head_index] = None
        self.head_index = (self.head_index + 1) % self.buffer_size
        if frame is not None:
            self.output_count += 1
        return frame

    def append(self, frame_id: int, data):
        index = frame_id % self.buffer_size
        self.buffer[index] = data
        self.last_frame_id = frame_id
        if self.head_index == -1:
            self.head_index = 0
        elif index == self.head_index:
            self.head_index += 1
            self.head_index %= self.buffer_size
    
    def flush(self) -> List:
        if self.head_index == -1:
            return []
            
        frames = []
        # 从head_index开始，按顺序取出所有有效帧
        current_index = self.head_index
        for _ in range(self.buffer_size):
            frame = self.buffer[current_index]
            if frame is not None:
                frames.append(frame)
                self.output_count += 1
            current_index = (current_index + 1) % self.buffer_size
            if current_index == self.head_index:
                break
                
        self.buffer = [None] * self.buffer_size
        self.head_index = -1
        return frames

# 一个视频裁剪函数，用于根据目标边界框从原始视频帧中裁剪出以目标为中心的固定大小区域
def cut_video(raw_frames: List[np.ndarray], bbox_list: List[Tuple], return_type="nd") -> Union[List[np.ndarray], np.ndarray]: 
    assert len(raw_frames) == len(bbox_list), f"{len(raw_frames)} != {len(bbox_list)}"
    bbox_list = [((b[0]+b[2])//2, (b[1]+b[3])//2, (b[2]-b[0]), (b[3]-b[1])) for b in bbox_list]
    max_width = int(max(b[2] for b in bbox_list)*2)
    max_height = int(max(b[3] for b in bbox_list)*2)
    max_size = max(max_width, max_height)
    max_width, max_height = max_size, max_size
    frame_list = []
    frame_id = 0
    center_list = [(b[0], b[1]) for b in bbox_list]
    center_list = smooth(center_list)
    for frame in raw_frames:
        x, y = center_list[frame_id]
        frame_width = frame.shape[1]
        frame_height = frame.shape[0]
        x1 = x - max_width/2
        y1 = y - max_height/2
        x2 = x + max_width/2
        y2 = y + max_height/2
        if x1 < 0:
            x1, x2 = 0, max_width
        if y1 < 0:
            y1, y2 = 0, max_height
        if x2 >= frame_width:
            x1, x2 = frame_width-max_width, frame_width
        if y2 >= frame_height:
            y1, y2 = frame_height-max_height, frame_height
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        frame = frame[y1:y2, x1:x2]
        frame_list.append(frame)
        frame_id += 1
    if return_type == "nd":
        return np.stack(frame_list)
    else:
        return frame_list

# 将视频分析结果导出到Excel的函数，用于统计工人行为的时间分布
def export_to_excel(res_by_frame, fps, excel_path):
    # 可配置参数
    ACTIVITY_NAMES = [
        "Clean", "Concrete", "Formwork", "Prepare", "Rebar",
        "RestTalk", "Scaffold", "Transport", "Walk"
    ]
    NUM_CLASSES = len(ACTIVITY_NAMES)
    
    frames = res_by_frame
    # workers[worker_id] -> [count_per_class...]
    workers = defaultdict(lambda: [0] * NUM_CLASSES)
    
    # 统计帧数
    for frame in frames:
        if not isinstance(frame, (list, tuple)):
            continue
        for det in frame:
            try:
                label = int(det[4])  # action_id
                wid = int(det[5])    # worker_id
            except (IndexError, ValueError, TypeError):
                continue
            if 0 <= label < NUM_CLASSES and wid >= 0:
                workers[wid][label] += 1
    
    # 计算总计行数据
    total_counts = [0] * NUM_CLASSES
    for wid in workers:
        for i in range(NUM_CLASSES):
            total_counts[i] += workers[wid][i]
    
    # 写 Excel
    with xlsxwriter.Workbook(excel_path) as wb:
        ws = wb.add_worksheet("sheet1")
        
        # 创建格式
        seconds_fmt = wb.add_format({"num_format": '0.0" s"'})
        
        # 写表头
        headers = ["Worker"] + ACTIVITY_NAMES + ["Total"]
        ws.write_row(0, 0, headers)
        
        # 写总计行
        total_seconds = [c / fps for c in total_counts]
        grand_total = sum(total_seconds)
        ws.write(1, 0, "Total")
        ws.write_row(1, 1, total_seconds, seconds_fmt)
        ws.write(1, 1 + NUM_CLASSES, grand_total, seconds_fmt)
        
        # 写每个工人的数据
        for row, wid in enumerate(sorted(workers), start=2):
            counts = workers[wid]
            seconds = [c / fps for c in counts]
            total = sum(seconds)
            
            ws.write(row, 0, f"Worker{wid}")
            ws.write_row(row, 1, seconds, seconds_fmt)
            ws.write(row, 1 + NUM_CLASSES, total, seconds_fmt)
    
    logging.info(f"Excel文件已保存到: {excel_path}")


def workflow(video_source, target_path, detect_model_path, action_model_path, device) -> dict:
    
    # 初始化变量
    input_streamer = None
    output_streamer = None

    try:
        device = torch.device(f"cuda:{device}")
        # 初始化输入流,获取视频信息
        input_streamer = InputStreamer(video_source, keep_rate=False)
        width = input_streamer.width
        height = input_streamer.height
        fps = input_streamer.fps
        frame_seconds = 1/fps
        # 初始化缓存系统
        frame_cache = FrameCache(maxsize=30)
        output_buffer = OutputBuffer(buffer_size=31)
        # 初始化输出流
        if target_path is None:
            output_streamer = OutputStreamer("dummy", width, height, fps)
        else:
            output_streamer = OutputStreamer(target_path, width, height, fps)
        
        # 当用户按Ctrl+C或系统请求终止时,关闭输入/输出流
        def signal_handler(sig, frame):
            input_streamer.terminate()
            output_streamer.terminate()
            logging.info("Received SIGINT or SIGTERM. Terminating gracefully...")
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # 模型:目标检测+动作识别+SORT
        # detector = YoloDetector(str(detect_model_path), device=device)
        detector = YoloDetector(detect_model_path, device=device)
        act_model = ActionRecognition(action_model_path, device=device)
        tracker = Sort(max_age=40, iou_threshold=0.1) # max_age,允许目标消失的最大帧数
        processing_started_at = time.perf_counter()

        bbox_buffers = collections.defaultdict(lambda: BBoxBuffer())
        last_action_id = {}
        frame_id0 = 0
        action_counter = collections.Counter()
        res_by_frame = []
        actual_input_frames = 0 # 记录实际输入的帧数
        
        for frame0 in input_streamer:
            actual_input_frames += 1  # 实时统计输入帧数
            res_by_frame.append([])
            frame_cache.append(frame0)

            # Release the oldest annotated frame before appending would wrap
            # around and overwrite it in the fixed-size circular buffer.
            if frame_id0 > 30:
                out_frame = output_buffer.pop()
                if out_frame is not None:
                    output_streamer.write(out_frame)

            output_buffer.append(frame_id0, copy.deepcopy(frame0))
            frame = copy.deepcopy(frame0)
            # list of detections, each detection is a numpy array, shape (N, 6), (x1, y1, x2, y2, score, class)
            det = detector.detect([frame], conf_thres=0.5)  # 目标检测
            det = det[0]
            if det.shape[0] > 0:
                # tr只包含同时满足：
                # 1.当前帧有检测匹配的（time_since_update < 1）
                # 2.且足够稳定的目标（hit_streak ≥ min_hits 或视频刚开始）
                # 目标跟踪,不要类别,更新tracker的self.track属性,返回当前帧可输出的目标
                tr = tracker.update(det[:,:5])  
            else:
                tr = tracker.update()
            for t in tr:
                tid = t[4]
                bbox_buffers[tid].append(t[:4].tolist(), frame_id0)
            
            cut_frames_list = []
            for tid in list(bbox_buffers.keys()):
                bbox_buffer = bbox_buffers[tid]
                if bbox_buffer.last_frame_id == frame_id0 and bbox_buffer.size < 30: # 缓冲区已满或目标消失,进行动作识别
                    continue
                else:
                    buffer = bbox_buffers.pop(tid)
                    bbox_list = buffer.buffer
                    first_frame_id = buffer.first_frame_id
                    last_frame_id = buffer.last_frame_id
                    raw_frames = []
                    for fid in range(first_frame_id, last_frame_id+1):
                        index = fid - frame_id0 - 1
                        frame = frame_cache.get(index)
                        raw_frames.append(frame)
                    cut_frames = cut_video(raw_frames, bbox_list)
                    # 
                    cut_frames_list.append({
                        "frames": cut_frames,   # 裁剪后的像素信息
                        "ffid": first_frame_id,
                        "tid": tid,
                        "bbox_list": bbox_list, # 包围框信息
                        "last_frame_id": last_frame_id
                        })
            for cut_frames in cut_frames_list:
                # 某个track_id在起始帧到结束帧的移动轨迹(bbox_list)
                frames = cut_frames["frames"] # 裁剪后的图像序列
                tid = cut_frames["tid"]       # 跟踪ID
                if frames.shape[0] < 10 and last_action_id.get(tid, -1) != -1:
                    act_id = last_action_id[cut_frames["tid"]]
                else:
                    res = act_model.detect(frames) # 动作识别
                    act_id = res["act_id"]
                    last_action_id[tid] = act_id
                ffid = cut_frames["ffid"] # 起始帧ID
                last_frame_id = cut_frames["last_frame_id"] # 结束帧ID
                bbox_list = cut_frames["bbox_list"] # 边界框序列
                action_counter[act_id] += len(frames)*frame_seconds
    
                for i in range(len(frames)):
                    out_frame = output_buffer.get(ffid+i)
                    if out_frame is not None:
                        box = bbox_list[i]
                        aid = act_id
                        draw(out_frame, box, tid, aid)
                        if ffid+i < len(res_by_frame):
                            res_by_frame[ffid+i].append((*[round(x, 3) for x in box], int(aid), int(tid)))
            
            frame_id0 += 1
    
        # 确保所有剩余帧都被输出
        for out_frame in output_buffer.flush():
            if out_frame is not None:
                output_streamer.write(out_frame)
    
        output_streamer.write(None)
        output_streamer.terminate()
        processing_seconds = time.perf_counter() - processing_started_at
        processing_fps = actual_input_frames / processing_seconds if processing_seconds else 0.0
        
        logging.info(f"实际输入帧数: {actual_input_frames}")
        logging.info(f"输出帧数: {output_buffer.output_count}")
        logging.info(f"端到端处理耗时: {processing_seconds:.3f} 秒")
        logging.info(f"端到端处理速度: {processing_fps:.3f} FPS")
        
        return {
            "summary": dict(action_counter),
            "res_by_frame": res_by_frame,
            "fps": fps,
            "实际输入帧数":actual_input_frames,
            "输出帧数":output_buffer.output_count,
            "处理耗时秒": processing_seconds,
            "处理速度FPS": processing_fps
        }
    finally:
        # 确保流被正确关闭
        if input_streamer:
            try:
                input_streamer.terminate()
                logging.info("Input streamer terminated")
            except Exception as e:
                logging.error(f"Error terminating input streamer: {e}")

        if output_streamer:
            try:
                output_streamer.terminate()
                logging.info("Output streamer terminated")
            except Exception as e:
                logging.error(f"Error terminating output streamer: {e}")


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--source", type=str, required=True, help="webcam or video path")
    arg_parser.add_argument("--target", type=str, help="target path")
    arg_parser.add_argument("--device", type=int, default=0, help="device id")
    arg_parser.add_argument("--result_path", type=str, help="result path, json file")
    arg_parser.add_argument("--excel_path", type=str, help="excel file path")
    arg_parser.add_argument(
        "--detect_model", type=Path,
        default=ROOT / "resource/det_action_old.onnx",
        help="ONNX or TensorRT engine used for worker detection",
    )
    arg_parser.add_argument(
        "--action_model", type=Path,
        default=ROOT / "resource/jit.end2end.pt",
        help="TorchScript model used for action recognition",
    )
    args = arg_parser.parse_args()

    # 模型权重路径
    detect_model_path = args.detect_model
    action_model_path = args.action_model
    # gpu
    device = args.device

    # 输入、输出
    video_source = args.source
    target_path = args.target
    result_path = args.result_path
    excel_path = args.excel_path
    
    check_res = workflow(
        video_source, 
        target_path, 
        detect_model_path, 
        action_model_path, 
        device
        )
    
    if args.result_path is not None:
        import json
        with open(result_path[7:], "w") as f:
            json.dump(check_res, f)
    if args.excel_path is not None:
        export_to_excel(check_res["res_by_frame"], check_res["fps"], excel_path[7:])
