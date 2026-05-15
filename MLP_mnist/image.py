#!/usr/bin/env python3
"""
Save a quantized MNIST image as a binary file (int8) suitable for the MLIR wrapper.

Usage:
    python save_quantized_image.py --json mnist_quantized.json [--output input.bin] [--index 0]

The script loads the quantized model from the JSON file to obtain the input scale.
It then fetches the specified MNIST test sample, normalizes it, quantizes it to int8,
and writes the 784 int8 values (row‑major, flat) to a binary file.

If torchvision is not available, it falls back to a zero image and prints a warning.
"""

import argparse
import json
import struct
import sys

def get_quantized_mnist_sample(json_path: str, sample_index: int = 0):
    """
    Load a quantized MNIST test sample using the input scale from the JSON model.
    Returns (int8_list, label)
    """
    # Load input scale from JSON
    with open(json_path) as f:
        data = json.load(f)
    scale_input = data["scales"]["input"]

    # Try to load a real MNIST image
    try:
        from torchvision import datasets, transforms
        import torch
    except ImportError:
        print("ERROR: torchvision not installed. Cannot load real MNIST image.", file=sys.stderr)
        sys.exit(1)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST("./data", train=False, download=True, transform=transform)
    img_tensor, label = dataset[sample_index]          # img_tensor: (1,28,28) float
    img_flat = img_tensor.view(-1)                     # (784,) float

    # Quantize to int8 using the model's input scale
    quantized = torch.round(img_flat / scale_input).clamp(-128, 127).to(torch.int8)
    return quantized.tolist(), label

def save_int8_binary(data: list, output_path: str):
    """
    Write a list of integers (each in -128..127) as raw int8 binary.
    """
    with open(output_path, "wb") as f:
        for value in data:
            # Convert to signed 8-bit byte (Python int -> bytes)
            f.write(struct.pack("b", value))

def main():
    parser = argparse.ArgumentParser(description="Save quantized MNIST image as int8 binary.")
    parser.add_argument("--json", required=True, help="Path to mnist_quantized.json (from train.py)")
    parser.add_argument("--output", default="input.bin", help="Output binary file (default: input.bin)")
    parser.add_argument("--index", type=int, default=2, help="MNIST test sample index (default: 0)")
    args = parser.parse_args()

    quantized_flat, label = get_quantized_mnist_sample(args.json, args.index)
    save_int8_binary(quantized_flat, args.output)

    print(f"Saved {len(quantized_flat)} int8 values (sample index {args.index}, label {label}) to {args.output}")
    print("First 20 values:", quantized_flat[:784])

if __name__ == "__main__":
    main()