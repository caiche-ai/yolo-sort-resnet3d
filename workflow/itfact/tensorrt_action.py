"""TensorRT runtime for the normalized 3D ResNet action core."""

from pathlib import Path
from typing import List

import numpy as np
import tensorrt as trt
import torch


class TensorRTActionRecognition:
    def __init__(self, engine_path, device=0):
        self.engine_path = Path(engine_path)
        self.device = torch.device(f"cuda:{device}")
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(
            self.engine_path.read_bytes()
        )
        if self.engine is None:
            raise RuntimeError(f"Unable to deserialize {self.engine_path}")
        self.context = self.engine.create_execution_context()
        self.input_index = next(
            index
            for index in range(self.engine.num_bindings)
            if self.engine.binding_is_input(index)
        )
        self.output_index = next(
            index
            for index in range(self.engine.num_bindings)
            if not self.engine.binding_is_input(index)
        )
        self.mean = torch.tensor(
            [0.4345, 0.4051, 0.3775],
            device=self.device,
            dtype=torch.float32,
        ).view(1, 3, 1, 1, 1)
        self.std = torch.tensor(
            [0.2768, 0.2713, 0.2737],
            device=self.device,
            dtype=torch.float32,
        ).view(1, 3, 1, 1, 1)

    def detect_batch(self, clips: List[List[np.ndarray]]) -> List[dict]:
        if not clips:
            return []
        inputs = np.stack([np.stack(clip) for clip in clips])
        if inputs.ndim != 5 or inputs.shape[-1] != 3:
            raise ValueError(f"Expected B,T,H,W,3 input, got {inputs.shape}")
        inputs = (
            torch.from_numpy(inputs)
            .permute(0, 4, 1, 2, 3)
            .to(device=self.device, dtype=torch.float32)
            .contiguous()
        )
        inputs = ((inputs / 255.0 - self.mean) / self.std).contiguous()
        shape = tuple(inputs.shape)
        if not self.context.set_binding_shape(self.input_index, shape):
            raise ValueError(f"Action input shape is outside engine profile: {shape}")
        output_shape = tuple(self.context.get_binding_shape(self.output_index))
        outputs = torch.empty(
            output_shape,
            device=self.device,
            dtype=torch.float32,
        )
        bindings = [0] * self.engine.num_bindings
        bindings[self.input_index] = inputs.data_ptr()
        bindings[self.output_index] = outputs.data_ptr()
        stream = torch.cuda.current_stream(self.device)
        if not self.context.execute_async_v2(bindings, stream.cuda_stream):
            raise RuntimeError("TensorRT action inference failed")
        probabilities = torch.softmax(outputs, dim=1).cpu()
        return [
            {
                "act_id": int(torch.argmax(result).item()),
                "logits": result,
            }
            for result in probabilities
        ]
