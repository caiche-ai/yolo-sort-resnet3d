import sys
from pathlib import Path
import cv2
import numpy as np
import torch
from typing import List, Tuple, Union
import argparse

ROOT = Path(__file__).parents[1]
ROOT not in sys.path and sys.path.append(ROOT) # add ROOT to PATH

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yield frame

class ActionRecognition(object):
    def __init__(self, model_path, device, input_size=240):
        self.model = torch.jit.load(str(model_path), map_location="cpu")
        self.model.eval()
        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        else:
            device = torch.device(device)
        self.device = device
        self.input_size = input_size
        self.model.to(self.device)
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        if self.dtype == torch.float16:
            self.model.half()

    def _resize_center_crop(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        if height <= 0 or width <= 0:
            raise ValueError(f"Invalid action frame shape: {frame.shape}")

        scale = self.input_size / min(height, width)
        resized_width = max(self.input_size, round(width * scale))
        resized_height = max(self.input_size, round(height * scale))
        frame = cv2.resize(
            frame,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )
        left = (resized_width - self.input_size) // 2
        top = (resized_height - self.input_size) // 2
        return np.ascontiguousarray(
            frame[top:top + self.input_size, left:left + self.input_size]
        )
    
    def detect(self, inputs: Union[List[np.ndarray], np.ndarray]):
        """
        Args:
            frames: list of np.ndarray, RGB, (0, 255)
        """
        return self.detect_batch([inputs])[0]

    def detect_batch(
        self,
        clips: List[Union[List[np.ndarray], np.ndarray]],
    ) -> List[dict]:
        """Run multiple equally sized RGB clips in one model invocation."""
        if not clips:
            return []

        batch = []
        for clip in clips:
            inputs = np.stack(
                [self._resize_center_crop(frame) for frame in clip]
            )
            if inputs.ndim != 4:
                raise ValueError(
                    f"Expected a 4D action clip, got shape {inputs.shape}"
                )
            inputs = torch.from_numpy(inputs)
            if inputs.shape[-1] == 3:  # (T, H, W, C)
                inputs = inputs.permute(3, 0, 1, 2)
            elif inputs.shape[1] == 3:  # (T, C, H, W)
                inputs = inputs.permute(1, 0, 2, 3)
            else:
                raise ValueError(
                    f"Unable to find RGB channel in action clip {inputs.shape}"
                )
            batch.append(inputs)

        inputs = torch.stack(batch).to(
            device=self.device,
            dtype=self.dtype,
        )
        with torch.no_grad():
            results = self.model(inputs)
        return [
            {
                "act_id": int(torch.argmax(result, dim=-1).item()),
                "logits": result,
            }
            for result in results
        ]

if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--source", type=str, required=True)
    arg_parser.add_argument("--device", type=int, default=0)
    args = arg_parser.parse_args()

    video_path = args.source
    device = args.device
    
    model_path  = ROOT / "resource/jit.end2end.pt"
    act_model = ActionRecognition(model_path, device=device)
    
    frames = [ frame for frame in read_video(video_path)][:20]
    det_res = act_model.detect(frames)
    print(det_res)





