#!/usr/bin/env python3
"""Export the fixed-size 3D ResNet action core to ONNX for TensorRT."""

import argparse
from pathlib import Path

import torch


def parse_args():
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=project_root / "workflow/resource/jit.end2end.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "workflow/resource/action_resnet3d.onnx",
    )
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument("--clip-length", type=int, default=30)
    parser.add_argument("--input-size", type=int, default=240)
    return parser.parse_args()


def main():
    args = parse_args()
    args.model = args.model.resolve()
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    scripted = torch.jit.load(str(args.model), map_location="cpu").eval()
    # Export only the ResNet core. Runtime preprocessing supplies normalized
    # N,C,T,H,W tensors; this avoids unsupported torchvision transforms in ONNX.
    model = scripted.model.eval()
    example = torch.zeros(
        (1, 3, args.clip_length, args.input_size, args.input_size),
        dtype=torch.float32,
    )
    torch.onnx.export(
        model,
        example,
        str(args.output),
        input_names=["normalized_clips"],
        output_names=["logits"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes={
            "normalized_clips": {0: "batch"},
            "logits": {0: "batch"},
        },
    )
    print(f"Exported {args.output}")


if __name__ == "__main__":
    main()
