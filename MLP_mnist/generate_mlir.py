"""
mlp_inference.py  –  Quantized MLP primitives + MLIR generation.

Supports two modes
------------------
1.  Built-in toy example  (4 → 8 → 4, hard-coded weights):
        python mlp_inference.py

2.  MNIST model produced by train.py:
        python mlp_inference.py --mnist mnist_quantized.json

Quantization data-flow (per layer)
------------------------------------
    scale_acc  = scale_input × scale_weight     (int32 accumulator unit)
    scale_out  = scale of the next layer's input
    requant    = scale_acc / scale_out           (int32 → int8 scale factor)

    b_int32  = round( b_float / scale_acc )
    h_int8   = clamp( round( acc_int32 × requant ), -128, 127 )
"""

import argparse
import json

import torch
from torch_mlir import torchscript


# ════════════════════════════════════════════════════════════════════════════
#  Primitives
# ════════════════════════════════════════════════════════════════════════════

class FullyConnectedInt8(torch.nn.Module):
    """
    Single quantized FC layer: int8 × int8 → int32.

    forward(x):  x @ weight.T + bias
    Shapes
    ------
    x       : (batch, in_features)       int8
    weight  : (out_features, in_features) int8  (stored transposed)
    bias    : (out_features,)             int32  [optional]
    output  : (batch, out_features)       int32
    """

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None = None):
        super().__init__()
        # Store transposed so the matmul is  x @ weight_t  (B,in)@(in,out)→(B,out)
        self.register_buffer("weight_t", weight.T.contiguous())   # (in, out) int8
        self.register_buffer("bias",     bias)                     # (out,)    int32

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch._int_mm(x, self.weight_t)      # (B, out) int32
        if self.bias is not None:
            out = out + self.bias
        return out


class ReLUInt32(torch.nn.Module):
    """ReLU for int32 tensors (torch.relu doesn't support integers)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, min=0)


class RequantizeToInt8(torch.nn.Module):
    """
    Requantize an int32 accumulator to int8 using a floating-point scale.

    Formula
    -------
        out_int8 = clamp( round( acc_int32 × scale ), -128, 127 )

    where  scale = scale_acc / scale_output
                 = (scale_input × scale_weight) / scale_output

    Args
    ----
    scale : float
        Requantization multiplier.  Pass 1.0 to get the legacy
        clamp-only behaviour (useful for the toy 4→8→4 example).
    """

    def __init__(self, scale: float = 1.0):
        super().__init__()
        # Registered as a buffer so it travels with the module (state_dict, MLIR).
        self.register_buffer(
            "scale", torch.tensor(scale, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast to float32, apply scale, round back, then clamp+cast to int8.
        scaled = torch.round(x.to(torch.float32) * self.scale).to(torch.int32)
        return torch.clamp(scaled, -128, 127).to(torch.int8)


# ════════════════════════════════════════════════════════════════════════════
#  3-Layer MLP  (InputLayer → HiddenLayer → OutputLayer)
# ════════════════════════════════════════════════════════════════════════════

class QuantizedMLP(torch.nn.Module):
    """
    3-layer quantized MLP.

    Data flow
    ---------
        x (int8)
        ──► FC1 ──► int32 ──► ReLU ──► RequantizeToInt8 ──► int8
        ──► FC2 ──► int32   (raw integer logits, argmax gives class)

    Args
    ----
    fc1, fc2      : FullyConnectedInt8 layers
    relu          : ReLUInt32
    requantize    : RequantizeToInt8  (carries the scale for FC1 output)
    scale_input   : float – scale of the int8 input  (for diagnostics / dequant)
    scale_acc2    : float – scale of FC2 int32 output (for diagnostics / dequant)
    """

    def __init__(
        self,
        fc1        : FullyConnectedInt8,
        fc2        : FullyConnectedInt8,
        relu       : ReLUInt32,
        requantize : RequantizeToInt8,
        scale_input: float = 1.0,
        scale_acc2 : float = 1.0,
    ):
        super().__init__()
        self.fc1        = fc1
        self.fc2        = fc2
        self.relu       = relu
        self.requantize = requantize
        # Stored as non-trainable scalars for dequantization / diagnostics.
        self.register_buffer("scale_input", torch.tensor(scale_input))
        self.register_buffer("scale_acc2",  torch.tensor(scale_acc2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ── Hidden layer ──────────────────────────────────────────────
        h = self.fc1(x)          # int8  → int32  (accumulate)
        h = self.relu(h)         # int32 → int32  (ReLU)
        h = self.requantize(h)   # int32 → int8   (scale + clamp + cast)

        # ── Output layer (no activation; caller applies argmax) ───────
        return self.fc2(h)       # int8  → int32


# ════════════════════════════════════════════════════════════════════════════
#  Factory helpers
# ════════════════════════════════════════════════════════════════════════════

def make_fc_layer(
    in_features : int,
    out_features: int,
    weight_data : list,
    bias_data   : list | None = None,
) -> FullyConnectedInt8:
    """
    Build a FullyConnectedInt8 from raw Python lists.

    Args
    ----
    weight_data : flat list, row-major (out_features × in_features), int8 range
    bias_data   : flat list, length out_features, int32 range  [optional]
    """
    weight = torch.tensor(weight_data, dtype=torch.int8).reshape(out_features, in_features)
    bias   = (
        torch.tensor(bias_data, dtype=torch.int32)
        if bias_data is not None else None
    )
    return FullyConnectedInt8(weight, bias).eval()


def make_mlp(
    layer_sizes   : list[int],
    weight_data   : list[list],
    bias_data     : list[list | None],
    requant_scale : float = 1.0,
    scale_input   : float = 1.0,
    scale_acc2    : float = 1.0,
) -> QuantizedMLP:
    """
    Build a QuantizedMLP from layer specs.

    Args
    ----
    layer_sizes   : [in_features, hidden_features, out_features]
    weight_data   : [w1_flat_list, w2_flat_list]   (int8 range)
    bias_data     : [b1_list_or_None, b2_list_or_None]  (int32 range)
    requant_scale : (scale_input × scale_w1) / scale_hidden
                   Multiplier applied to FC1's int32 output before casting to int8.
                   Use 1.0 for the legacy clamp-only behaviour.
    scale_input   : scale of the int8 model input (for diagnostics)
    scale_acc2    : scale of FC2's int32 output   (for diagnostics / dequant)
    """
    assert len(layer_sizes) == 3, "layer_sizes must be [in, hidden, out]"
    assert len(weight_data) == 2, "Need 2 weight matrices"
    assert len(bias_data)   == 2, "Need 2 bias entries (or None)"

    n_in, n_hidden, n_out = layer_sizes

    fc1 = make_fc_layer(n_in,     n_hidden, weight_data[0], bias_data[0])
    fc2 = make_fc_layer(n_hidden, n_out,    weight_data[1], bias_data[1])

    return QuantizedMLP(
        fc1, fc2,
        ReLUInt32(),
        RequantizeToInt8(scale=requant_scale),
        scale_input=scale_input,
        scale_acc2=scale_acc2,
    ).eval()


# ════════════════════════════════════════════════════════════════════════════
#  Load from JSON produced by train.py
# ════════════════════════════════════════════════════════════════════════════

def load_from_json(path: str) -> tuple[QuantizedMLP, dict]:
    """
    Deserialize a quantized model saved by train.py.

    Returns
    -------
    (model, meta)
        model : QuantizedMLP ready for inference and MLIR compilation
        meta  : dict with layer_sizes and the full scales sub-dict
    """
    with open(path) as f:
        data = json.load(f)

    scales = data["scales"]

    model = make_mlp(
        layer_sizes   = data["layer_sizes"],
        weight_data   = [data["w1"], data["w2"]],
        bias_data     = [data["b1"], data["b2"]],
        requant_scale = scales["requant_fc1"],
        scale_input   = scales["input"],
        scale_acc2    = scales["acc2"],
    )
    return model, data


# ════════════════════════════════════════════════════════════════════════════
#  MLIR generation
# ════════════════════════════════════════════════════════════════════════════

def generate_mlp_mlir(
    model       : QuantizedMLP,
    layer_sizes : list[int],
    input_data  : list,
    batch_size  : int = 1,
    output_file : str = "mlp.mlir",
    scales      : dict | None = None,
) -> str:
    """
    Compile a QuantizedMLP to MLIR (linalg-on-tensors) and write it to disk.

    Args
    ----
    model       : QuantizedMLP (already .eval())
    layer_sizes : [in_features, hidden_features, out_features]
    input_data  : flat list of int8 values, length = batch_size × in_features
    batch_size  : number of input samples in the compilation trace
    output_file : path for the generated .mlir file
    scales      : optional dict of float scales for the diagnostic printout
    """
    n_in = layer_sizes[0]
    x = torch.tensor(input_data, dtype=torch.int8).reshape(batch_size, n_in)

    # ── Ground-truth forward pass ────────────────────────────────────────
    with torch.no_grad():
        expected = model(x)

    # ── torch-mlir compilation ───────────────────────────────────────────
    mlir_module = torchscript.compile(
        model,
        (x,),
        output_type=torchscript.OutputType.LINALG_ON_TENSORS,
    )
    mlir_text = str(mlir_module)

    with open(output_file, "w") as f:
        f.write(mlir_text)

    # ── Diagnostics ──────────────────────────────────────────────────────
    sep = "=" * 60
    print(sep)
    print(f"MLP architecture  : {' → '.join(str(n) for n in layer_sizes)}")
    print(f"Batch size        : {batch_size}")
    print(f"Output file       : {output_file}")
    if scales:
        print(sep)
        print("Quantization scales:")
        for k, v in scales.items():
            print(f"  {k:<14} = {v:.8f}")
    print(sep)
    print(f"\nInput x  ({batch_size}×{n_in}, int8):\n{x}")
    print(f"\nExpected output ({batch_size}×{layer_sizes[-1]}, int32):\n{expected}")
    if scales and "acc2" in scales:
        dequant = expected.float() * scales["acc2"]
        print(f"\nDequantized output (float32, ×scale_acc2={scales['acc2']:.6f}):\n{dequant}")
    print(f"\n── MLIR (first 2 000 chars) ──\n{mlir_text[:2000]} …")

    return mlir_text


# ════════════════════════════════════════════════════════════════════════════
#  Example 1 – Toy MLP  4 → 8 → 4
# ════════════════════════════════════════════════════════════════════════════

def run_toy_example() -> None:
    """
    Built-in toy demo with hand-crafted int8 weights.
    requant_scale=1.0  ⟹  legacy clamp-only requantization.
    """
    layer_sizes = [4, 8, 4]

    # FC1: shape (8 out × 4 in)
    w1 = [
         1,  2,  1,  2,
         2,  1,  2,  1,
         1, -1,  1, -1,
        -1,  2, -1,  2,
         3,  1,  0,  1,
         0,  2,  3,  2,
         1,  1,  1,  1,
         2, -1,  2, -1,
    ]

    # FC2: shape (4 out × 8 in)
    w2 = [
         1,  2,  1,  0,  1,  2,  1,  0,
         0,  1,  2,  1,  0,  1,  2,  1,
         1,  0,  1,  2,  1,  0,  1,  2,
         2,  1,  0,  1,  2,  1,  0,  1,
    ]

    b1 = [1, 2, 1, 2, 1, 2, 1, 2]
    b2 = [10, 20, 10, 20]
    x  = [3, 1, 4, 2]

    mlp = make_mlp(
        layer_sizes   = layer_sizes,
        weight_data   = [w1, w2],
        bias_data     = [b1, b2],
        requant_scale = 1.0,    # no scale – legacy clamp-only
    )

    generate_mlp_mlir(
        model       = mlp,
        layer_sizes = layer_sizes,
        input_data  = x,
        batch_size  = 1,
        output_file = "mlp_4x8x4.mlir",
    )


# ════════════════════════════════════════════════════════════════════════════
#  Example 2 – MNIST MLP  784 → 128 → 10  (loaded from train.py output)
# ════════════════════════════════════════════════════════════════════════════

def run_mnist_example(json_path: str) -> None:
    """
    Load the MNIST quantized model from json_path and compile it to MLIR.

    A single all-zeros int8 input sample is used as the compilation trace;
    the important thing is the tensor shape, not the values.  Swap in a real
    MNIST sample by replacing `x_sample` below.
    """
    print(f"Loading quantized MNIST model from '{json_path}' …")
    model, data = load_from_json(json_path)
    layer_sizes  = data["layer_sizes"]          # [784, 128, 10]
    scales       = data["scales"]

    n_in         = layer_sizes[0]               # 784

    # ── Use a real (quantized) MNIST sample if torchvision is available ──
    try:
        from torchvision import datasets, transforms
        _tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        ds     = datasets.MNIST("./data", train=False, download=True, transform=_tf)
        img, label = ds[0]                                          # (1,28,28)
        img_flat   = img.view(-1)                                   # (784,) float
        scale_in   = scales["input"]
        x_int8     = torch.clamp(
            torch.round(img_flat / scale_in), -128, 127
        ).to(torch.int8)
        x_sample   = x_int8.tolist()
        print(f"Using MNIST test sample 0  (label = {label})")
    except Exception:
        # Fallback: all-zeros trace input
        x_sample = [0] * n_in
        print("(torchvision unavailable – using zero input for MLIR trace)")

    generate_mlp_mlir(
        model       = model,
        layer_sizes = layer_sizes,
        input_data  = x_sample,
        batch_size  = 1,
        output_file = "mlp_mnist_784x128x10.mlir",
        scales      = scales,
    )


# ════════════════════════════════════════════════════════════════════════════
#  CLI entry point
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate MLIR for a quantized int8 MLP."
    )
    parser.add_argument(
        "--mnist",
        metavar="JSON",
        default=None,
        help=(
            "Path to the JSON file produced by train.py "
            "(e.g. mnist_quantized.json).  "
            "If omitted, the built-in toy 4→8→4 example is run."
        ),
    )
    args = parser.parse_args()

    if args.mnist:
        run_mnist_example(args.mnist)
    else:
        run_toy_example()


if __name__ == "__main__":
    main()