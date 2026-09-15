import torch
import numpy as np
from pathlib import Path
from .util import letterbox, NMS, scale_boxes, YoloPreprocess
from torchvision.transforms import Compose, ToTensor, Normalize
import logging
import time
from typing import List
import onnxruntime


class AverageTimeCostManager(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.time_cost = {}
        self.count = {}

    def update(self, time_cost):
        for k, v in time_cost.items():
            if k not in self.time_cost:
                self.time_cost[k] = 0
                self.count[k] = 0
            self.time_cost[k] += v
            self.count[k] += 1
    
    def get(self):
        res = {}
        for k, v in self.time_cost.items():
            res[k] = v / self.count[k]
        return res

class YoloDetector(object):
    def __init__(self, weights_path, device='cpu', calc_time_cost=False, batch_size=1, imgsz=640):
        weights_path = Path(weights_path)
        self.device = device
        self.cuda = self.device != 'cpu'
        self.half = self.device != 'cpu'
        self.onnx = False
        self.tensorrt = False
        if weights_path.suffix.lower() == ".onnx":
            self._init_onnx(weights_path)
        elif weights_path.suffix.lower() == ".engine":
            self._init_tensorrt(weights_path)
        else:
            raise NotImplementedError(f"Unsupported detector model: {weights_path}")
        self._transform = Compose([
            ToTensor(),
        ])
        self.batch_size = batch_size
        self.imgsz = imgsz
        self.calc_time_cost = calc_time_cost
        self.time_cost_manager = AverageTimeCostManager()
        #self.preprocessor = YoloPreprocess(target_shape=imgsz, stride=self.stride, auto=False).to(self.device)
        logging.debug(f"YoloDetector initialized on {self.device}")
        logging.debug(f"Model: {weights_path}")
        self._warmup()


    def _preprocess1(self, imgs: List[np.array], imgsz=640):
        inputs = torch.from_numpy(np.stack(imgs)).to(self.device).transpose(1, 3)
        inputs, _, _ = self.preprocessor(inputs)
        return list(inputs.cpu().numpy())

    def _preprocess(self, imgs, imgsz=640):
        """
        Args:
            imgs (list): list of images, each image is a numpy array, shape (H, W, C), RGB
            imgsz (int): image size

        Returns:
            inputs (list): list of images, each image is a numpy array, shape (C, H, W), RGB
        """
        inputs = []
        for img in imgs:
            img = letterbox(img, new_shape=imgsz, auto=False, stride=self.stride)[0]
            #img = self._transform(img)
            #img = img.numpy()
            img = img.transpose(2, 0, 1) # HWC to CHW
            img = np.ascontiguousarray(img)
            img = img.astype(np.float32)
            img /= 255.0
            inputs.append(img)
        return inputs

    def _init_onnx(self, weights_path):
        
        self.onnx = True
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.cuda else ["CPUExecutionProvider"]

        self.session = onnxruntime.InferenceSession(weights_path, providers=providers)
        metadata = self.session.get_modelmeta().custom_metadata_map
        self.stride = int(metadata['stride'])
        self.output_names = [x.name for x in self.session.get_outputs()]

    def _init_tensorrt(self, weights_path):
        if not self.cuda:
            raise ValueError("TensorRT engines require a CUDA device")

        try:
            import tensorrt as trt
        except ImportError as exc:
            raise RuntimeError(
                "TensorRT Python bindings are unavailable. On Jetson with a Conda "
                "environment, add /usr/lib/python3.8/dist-packages to PYTHONPATH."
            ) from exc

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(weights_path, "rb") as engine_file:
            engine = runtime.deserialize_cuda_engine(engine_file.read())
        if engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {weights_path}")

        context = engine.create_execution_context()
        if context is None:
            raise RuntimeError(f"Failed to create TensorRT context: {weights_path}")

        input_indices = [
            index for index in range(engine.num_bindings)
            if engine.binding_is_input(index)
        ]
        output_indices = [
            index for index in range(engine.num_bindings)
            if not engine.binding_is_input(index)
        ]
        if len(input_indices) != 1:
            raise RuntimeError(
                f"Expected one TensorRT input, found {len(input_indices)}"
            )

        self.tensorrt = True
        self.trt = trt
        self.trt_logger = logger
        self.trt_runtime = runtime
        self.engine = engine
        self.context = context
        self.input_index = input_indices[0]
        self.output_indices = output_indices
        self.output_names = [engine.get_binding_name(i) for i in output_indices]
        # Letterbox uses auto=False, so the exact stride does not affect padding.
        self.stride = 32



    def _from_numpy(self, x):
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x

    def _trt_to_torch_dtype(self, dtype):
        mapping = {
            self.trt.float32: torch.float32,
            self.trt.float16: torch.float16,
            self.trt.int8: torch.int8,
            self.trt.int32: torch.int32,
            self.trt.bool: torch.bool,
        }
        try:
            return mapping[dtype]
        except KeyError as exc:
            raise TypeError(f"Unsupported TensorRT binding dtype: {dtype}") from exc

    def _forward(self, im):
        if self.onnx:  # ONNX Runtime
            if isinstance(im, torch.Tensor):
                im = im.cpu().numpy()  # torch to numpy
            y = self.session.run(self.output_names, {self.session.get_inputs()[0].name: im})
        elif self.tensorrt:
            input_dtype = self._trt_to_torch_dtype(
                self.engine.get_binding_dtype(self.input_index)
            )
            if not isinstance(im, torch.Tensor):
                im = torch.as_tensor(np.asarray(im), device=self.device)
            im = im.to(device=self.device, dtype=input_dtype).contiguous()

            if not self.context.set_binding_shape(self.input_index, tuple(im.shape)):
                raise RuntimeError(
                    f"TensorRT engine rejected input shape {tuple(im.shape)}"
                )

            bindings = [0] * self.engine.num_bindings
            bindings[self.input_index] = im.data_ptr()
            outputs = []
            for output_index in self.output_indices:
                output_shape = tuple(self.context.get_binding_shape(output_index))
                output_dtype = self._trt_to_torch_dtype(
                    self.engine.get_binding_dtype(output_index)
                )
                output = torch.empty(
                    output_shape, dtype=output_dtype, device=self.device
                )
                bindings[output_index] = output.data_ptr()
                outputs.append(output)

            stream = torch.cuda.current_stream(device=self.device)
            if not self.context.execute_async_v2(
                bindings=bindings, stream_handle=stream.cuda_stream
            ):
                raise RuntimeError("TensorRT inference execution failed")
            y = outputs[0] if len(outputs) == 1 else outputs
        if isinstance(y, list):
            y = self._from_numpy(y[0]) if len(y) == 1 else [self._from_numpy(a) for a in y]
        else:
            y = self._from_numpy(y)  
        return y
    
    def _warmup(self):
        inputs = self._preprocess([np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)]*self.batch_size, imgsz=self.imgsz)
        self._forward(inputs)

    def detect(self, imgs, conf_thres=0.25, iou_thres=0.45):
        """
        imgs (list): list of images, each image is a numpy array, shape (H, W, C), RGB
        return (list): list of detections, each detection is a numpy array, shape (N, 6), (x1, y1, x2, y2, score, class)
        """
        ts1 = time.time()
        time_cost = {}
        inputs = self._preprocess(imgs, self.imgsz)
        ts2 = time.time()
        ts1, time_cost['preprocess'] = ts2, ts2 - ts1
        pred = self._forward(inputs)
        ts2 = time.time()
        ts1, time_cost['inference'] = ts2, ts2 - ts1
        pred = NMS(pred, conf_thres, iou_thres, classes=None, agnostic=False)
        ts2 = time.time()
        ts1, time_cost['NMS'] = ts2, ts2 - ts1
        # Process predictions
        dets = []
        for i, det in enumerate(pred):  # per image
            det = det.cpu().detach().numpy()
            im0_shape = imgs[i].shape
            im_shape = inputs[i].shape
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im_shape[1:], det[:, :4], im0_shape).round()
            dets.append(det)
        ts2 = time.time()
        ts1, time_cost['postprocess'] = ts2, ts2 - ts1
        if self.calc_time_cost:
            logging.debug((
                f"Preprocess: {time_cost['preprocess']:.3f},"
                f"Inference: {time_cost['inference']:.3f},"
                f"NMS: {time_cost['NMS']:.3f},"
                f"Postprocess: {time_cost['postprocess']:.3f}"
                ))
            self.time_cost_manager.update(time_cost)
        return dets
