


import argparse
import json

import torch
from torch_mlir import torchscript


# ════════════════════════════════════════════════════════════════════════════
#  Shared primitives  (unchanged from MLP version)
# ════════════════════════════════════════════════════════════════════════════

class FullyConnectedInt8(torch.nn.Module):
    """
    Single quantized FC layer: int8 × int8 → int32.

    forward(x):  x @ weight.T + bias
    Shapes
    ------
    x       : (batch, in_features)        int8
    weight  : (out_features, in_features) int8  (stored transposed)
    bias    : (out_features,)             int32  [optional]
    output  : (batch, out_features)       int32
    """

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None = None):
        super().__init__()
        # Store transposed so the matmul is  x @ weight_t  (B,in)@(in,out)→(B,out)
        self.register_buffer("weight_t", weight.T.contiguous())  # (in, out) int8
        self.register_buffer("bias",     bias)                    # (out,)    int32

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch._int_mm(x, self.weight_t)    # (B, out) int32
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
        out_int8 = clamp(round(acc_int32 × scale), -128, 127)

    where  scale = scale_acc / scale_output
                 = (scale_input × scale_weight) / scale_output

    Works for any shape: scalars, 2-D (B, C), or 4-D (N, C, H, W).
    """

    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.register_buffer(
            "scale", torch.tensor(scale, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scaled = torch.round(x.to(torch.float32) * self.scale).to(torch.int32)
        return torch.clamp(scaled, -128, 127).to(torch.int8)


# ════════════════════════════════════════════════════════════════════════════
#  New conv primitives  (for LeNet only)
# ════════════════════════════════════════════════════════════════════════════

class Im2ColConvInt8(torch.nn.Module):
    """
    Quantized 2-D convolution expressed as slice-gather + torch._int_mm.

    Forward: (N, C_in, H, W) int8 → (N, C_out, H_out, W_out) int32

    Weight storage
    --------------
    weight_t : (C_in*kH*kW, C_out) int8
        Pre-transposed so the matmul is x_col @ weight_t.

    Bias
    ----
    bias : (C_out,) int32  [optional]
        Added to every row of the (N*H_out*W_out, C_out) output before
        reshaping, equivalent to a per-channel spatial bias in conv output.
    """

    def __init__(
        self,
        weight  : torch.Tensor,               # (C_out, C_in, kH, kW) int8
        bias    : torch.Tensor | None = None,  # (C_out,) int32
        in_h    : int = 28,
        in_w    : int = 28,
        stride  : int = 1,
        padding : int = 0,
        gemmini_pad_conv : bool = False,
        gemmini_chunk_rows : int = 0,
    ):
        super().__init__()
        self.C_out, self.C_in, self.kH, self.kW = weight.shape
        self.in_h = in_h
        self.in_w = in_w
        self.stride  = stride
        self.padding = padding
        # weight_t: (K, N) int8 — pre-transposed for matmul
        weight_t = weight.reshape(self.C_out, -1).T.contiguous()
        k = weight_t.shape[0]
        n = weight_t.shape[1]
        self.gemmini_pad_conv = gemmini_pad_conv
        self.gemmini_chunk_rows = gemmini_chunk_rows
        if gemmini_pad_conv:
            # Pad (K, N) to Gemmini-friendly multiples of 16 so lowering avoids
            # problematic tail paths on some FPGA configs.
            pad_unit = 16
            self.K_pad = ((k + pad_unit - 1) // pad_unit) * pad_unit
            self.N_pad = ((n + pad_unit - 1) // pad_unit) * pad_unit
            weight_t_padded = torch.zeros((self.K_pad, self.N_pad), dtype=torch.int8)
            weight_t_padded[:k, :n] = weight_t
            self.register_buffer("weight_t", weight_t_padded)
        else:
            self.K_pad = k
            self.N_pad = n
            self.register_buffer("weight_t", weight_t)
        self.register_buffer("bias", bias)
        h_eff = in_h + 2 * padding
        w_eff = in_w + 2 * padding
        self.H_out = (h_eff - self.kH) // stride + 1
        self.W_out = (w_eff - self.kW) // stride + 1
        k = self.C_in * self.kH * self.kW
        idx = []
        for oy in range(self.H_out):
            for ox in range(self.W_out):
                for ci in range(self.C_in):
                    for ky in range(self.kH):
                        for kx in range(self.kW):
                            iy = oy * stride + ky
                            ix = ox * stride + kx
                            idx.append(ci * h_eff * w_eff + iy * w_eff + ix)
                if self.gemmini_pad_conv:
                    # Pad each im2col row to K_pad with dummy index 0. These
                    # terms multiply against zero-padded weights.
                    for _ in range(self.K_pad - k):
                        idx.append(0)
        self.register_buffer("gather_idx", torch.tensor(idx, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Keep compile trace fully static (batch=1) to avoid dynamic-rank
        # shape propagation bugs in current torch-mlir builds.
        N = 1
        if self.padding != 0:
            x = torch.nn.functional.pad(
                x, (self.padding, self.padding, self.padding, self.padding)
            )
        x_flat = x.contiguous().view(1, -1)
        cols = x_flat.index_select(1, self.gather_idx)
        total_rows = self.H_out * self.W_out
        cols = cols.view(total_rows, self.K_pad)
        if self.gemmini_pad_conv and self.gemmini_chunk_rows > 0 and total_rows > self.gemmini_chunk_rows:
            # Avoid Python list<tensor> construction so Torch-MLIR can lower.
            out_padded = torch.zeros(
                (total_rows, self.N_pad), dtype=torch.int32, device=cols.device
            )
            for start in range(0, total_rows, self.gemmini_chunk_rows):
                end = min(start + self.gemmini_chunk_rows, total_rows)
                out_padded[start:end, :] = torch._int_mm(cols[start:end, :], self.weight_t)
        else:
            out_padded = torch._int_mm(cols, self.weight_t)
        out = out_padded[:, : self.C_out]
        out_nhwc = out.view(1, self.H_out * self.W_out, self.C_out)
      
        out = out_nhwc.permute(0, 2, 1).contiguous().view(1, self.C_out, self.H_out, self.W_out)
        return out


class MaxPool2dInt8(torch.nn.Module):
    """
    Max-pool for int8 feature maps, via a float32 round-trip.

    torch.max_pool2d requires float input; the cast is exact because
    every int8 value is exactly representable as float32.
    """

    def __init__(self, kernel_size: int = 2, stride: int = 2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride      = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.nn.functional.max_pool2d(
            x.to(torch.float32), self.kernel_size, self.stride
        )
        return out.to(torch.int8)


# ════════════════════════════════════════════════════════════════════════════
#  LeNet model
# ════════════════════════════════════════════════════════════════════════════

class QuantizedLeNet(torch.nn.Module):
    """
    Quantized LeNet-5 for MNIST.

    forward() accepts (N, 784) int8.
    The reshape to (N, 1, 28, 28) happens inside forward().
    ─────────────────────────────────────────────────────────────────────────

    Data flow
    ---------
    (N, 784) int8
      → view(N, 1, 28, 28)
      → Im2ColConv1 [1→6,  5×5] → (N,6,24,24) int32
      → ReLU → RequantizeToInt8 → MaxPool2d    → (N,6,12,12) int8
      → Im2ColConv2 [6→16, 5×5] → (N,16,8,8)  int32
      → ReLU → RequantizeToInt8 → MaxPool2d    → (N,16,4,4) int8
      → view(N, 256)
      → FC1  [256→120] → int32 → ReLU → RequantizeToInt8 → int8
      → FC2  [120→84]  → int32 → ReLU → RequantizeToInt8 → int8
      → FC3  [84→10]   → int32   ← logits (argmax → predicted digit)
    """

    def __init__(
        self,
        conv1 : Im2ColConvInt8,
        conv2 : Im2ColConvInt8,
        fc1   : FullyConnectedInt8,
        fc2   : FullyConnectedInt8,
        fc3   : FullyConnectedInt8,
        relu  : ReLUInt32,
        pool  : MaxPool2dInt8,
        rq1   : RequantizeToInt8,   # after conv1
        rq2   : RequantizeToInt8,   # after conv2
        rq3   : RequantizeToInt8,   # after fc1
        rq4   : RequantizeToInt8,   # after fc2
    ):
        super().__init__()
        self.conv1 = conv1
        self.conv2 = conv2
        self.fc1   = fc1
        self.fc2   = fc2
        self.fc3   = fc3
        self.relu  = relu
        self.pool  = pool
        self.rq1   = rq1
        self.rq2   = rq2
        self.rq3   = rq3
        self.rq4   = rq4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 784) int8
        N = 1

        # ── Reshape flat input to image ───────────────────────────────────
        h = x.view(1, 1, 28, 28)                   # (1, 1, 28, 28) int8

        # ── Conv block 1 ──────────────────────────────────────────────────
        h = self.conv1(h)                           # (N, 6, 24, 24) int32
        h = self.relu(h)                            # int32 ReLU
        h = self.rq1(h)                             # (N, 6, 24, 24) int8
        h = self.pool(h)                            # (N, 6, 12, 12) int8

        # ── Conv block 2 ──────────────────────────────────────────────────
        h = self.conv2(h)                           # (N, 16, 8, 8) int32
        h = self.relu(h)
        h = self.rq2(h)                             # (N, 16, 8, 8) int8
        h = self.pool(h)                            # (N, 16, 4, 4) int8

        # ── Flatten ───────────────────────────────────────────────────────
        h = h.view(1, 256)                          # (1, 256) int8

        # ── FC block 1 ────────────────────────────────────────────────────
        h = self.fc1(h)                             # (N, 120) int32
        h = self.relu(h)
        h = self.rq3(h)                             # (N, 120) int8

        # ── FC block 2 ────────────────────────────────────────────────────
        h = self.fc2(h)                             # (N, 84) int32
        h = self.relu(h)
        h = self.rq4(h)                             # (N, 84) int8

        # ── Output layer (no activation) ──────────────────────────────────
        return self.fc3(h)                          # (N, 10) int32


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

    weight_data : flat list, row-major (out_features × in_features), int8 range
    bias_data   : flat list, length out_features, int32 range  [optional]
    """
    weight = torch.tensor(weight_data, dtype=torch.int8).reshape(out_features, in_features)
    bias   = (
        torch.tensor(bias_data, dtype=torch.int32)
        if bias_data is not None else None
    )
    return FullyConnectedInt8(weight, bias).eval()


def make_conv_layer(
    C_out       : int,
    C_in        : int,
    kH          : int,
    kW          : int,
    weight_data : list,              # flat, row-major (C_out, C_in, kH, kW), int8 range
    bias_data   : list | None = None,  # flat, length C_out, int32 range
    stride      : int = 1,
    padding     : int = 0,
    in_h        : int = 28,
    in_w        : int = 28,
    gemmini_pad_conv : bool = False,
    gemmini_chunk_rows : int = 0,
) -> Im2ColConvInt8:
    """
    Build an Im2ColConvInt8 from raw Python lists.

    Mirrors make_fc_layer() but for 4-D conv weights.
    """
    weight = torch.tensor(weight_data, dtype=torch.int8).reshape(C_out, C_in, kH, kW)
    bias   = (
        torch.tensor(bias_data, dtype=torch.int32)
        if bias_data is not None else None
    )
    return Im2ColConvInt8(
        weight, bias, in_h=in_h, in_w=in_w, stride=stride, padding=padding,
        gemmini_pad_conv=gemmini_pad_conv, gemmini_chunk_rows=gemmini_chunk_rows
    ).eval()


# ════════════════════════════════════════════════════════════════════════════
#  LeNet factory
# ════════════════════════════════════════════════════════════════════════════

def make_lenet(
    data: dict,
    gemmini_pad_conv: bool = True,
    gemmini_chunk_rows: int = 0,
) -> QuantizedLeNet:
    """
    Build a QuantizedLeNet from a lenet_quantized.json dict.

    Build QuantizedLeNet from a lenet_quantized.json dict.
    """
    scales = data["scales"]

    conv1 = make_conv_layer(
        6, 1, 5, 5, data["wc1"], data["bc1"], in_h=28, in_w=28,
        gemmini_pad_conv=gemmini_pad_conv, gemmini_chunk_rows=gemmini_chunk_rows
    )
    conv2 = make_conv_layer(
        16, 6, 5, 5, data["wc2"], data["bc2"], in_h=12, in_w=12,
        gemmini_pad_conv=gemmini_pad_conv, gemmini_chunk_rows=gemmini_chunk_rows
    )

    # FC layers — identical call to make_fc_layer as in the MLP factory
    fc1   = make_fc_layer(256, 120, data["wf1"], data["bf1"])
    fc2   = make_fc_layer(120,  84, data["wf2"], data["bf2"])
    fc3   = make_fc_layer( 84,  10, data["wf3"], data["bf3"])

    return QuantizedLeNet(
        conv1=conv1, conv2=conv2,
        fc1=fc1,     fc2=fc2,     fc3=fc3,
        relu=ReLUInt32(),
        pool=MaxPool2dInt8(kernel_size=2, stride=2),
        rq1=RequantizeToInt8(scales["requant_c1"]),
        rq2=RequantizeToInt8(scales["requant_c2"]),
        rq3=RequantizeToInt8(scales["requant_f1"]),
        rq4=RequantizeToInt8(scales["requant_f2"]),
    ).eval()


# ════════════════════════════════════════════════════════════════════════════
#  JSON loader
# ════════════════════════════════════════════════════════════════════════════

def load_lenet_from_json(
    path: str,
    gemmini_pad_conv: bool = True,
    gemmini_chunk_rows: int = 0,
) -> tuple[QuantizedLeNet, dict]:
    """Load a QuantizedLeNet saved by train.py."""
    with open(path) as f:
        data = json.load(f)
    model = make_lenet(
        data,
        gemmini_pad_conv=gemmini_pad_conv,
        gemmini_chunk_rows=gemmini_chunk_rows,
    )
    return model, data


# ════════════════════════════════════════════════════════════════════════════
#  MLIR generation
# ════════════════════════════════════════════════════════════════════════════

def _compile_to_mlir(
    model       : torch.nn.Module,
    x           : torch.Tensor,
    output_file : str,
) -> str:
    """
    Common torch-mlir compilation step.
    """
    mlir_module = torchscript.compile(
        model,
        (x,),
        output_type=torchscript.OutputType.LINALG_ON_TENSORS,
    )
    mlir_text = str(mlir_module)
    with open(output_file, "w") as f:
        f.write(mlir_text)
    return mlir_text


def generate_lenet_mlir(
    model       : QuantizedLeNet,
    input_data  : list,              # flat int8 list, length 784
    batch_size  : int = 1,
    output_file : str = "lenet.mlir",
    scales      : dict | None = None,
) -> str:
    """
    Compile a QuantizedLeNet to MLIR and write it to disk.

    The trace input is (batch_size, 784) int8.
    """
    x = torch.tensor(input_data, dtype=torch.int8).reshape(batch_size, 784)

    with torch.no_grad():
        expected = model(x)

    mlir_text = _compile_to_mlir(model, x, output_file)

    sep = "=" * 60
    print(sep)
    print("LeNet-5  :  1×28×28  →  C1(6,5×5) pool  →  C2(16,5×5) pool")
    print("         →  FC(256→120)  →  FC(120→84)  →  FC(84→10)")
    print(f"Batch size        : {batch_size}")
    print(f"Output file       : {output_file}")
    if scales:
        print(sep)
        print("Quantization scales:")
        for k, v in scales.items():
            print(f"  {k:<14} = {v:.8f}")
    print(sep)
    print(f"\nInput shape: ({batch_size}, 784) int8  [reshaped to ({batch_size},1,28,28) inside model]")
    print(f"\nExpected output ({batch_size}×10, int32):\n{expected}")
    print(f"  → Predicted digit: {int(expected.argmax(dim=1).item())}")
    print(f"\n── MLIR (first 2 000 chars) ──\n{mlir_text[:2000]} …")
    return mlir_text


# ════════════════════════════════════════════════════════════════════════════
#  LeNet MNIST
# ════════════════════════════════════════════════════════════════════════════

def run_lenet_example(
    json_path: str,
    gemmini_pad_conv: bool = True,
    gemmini_chunk_rows: int = 0,
    output_file: str = "lenet_mnist.mlir",
) -> None:
    """
    Load the quantized LeNet from json_path and compile it to MLIR.

    The trace input is (1, 784) int8 — first MNIST test image quantized
    with scale_input from the JSON.
    """
    print(f"Loading quantized LeNet from '{json_path}' …")
    model, data = load_lenet_from_json(
        json_path,
        gemmini_pad_conv=gemmini_pad_conv,
        gemmini_chunk_rows=gemmini_chunk_rows,
    )
    scales      = data["scales"]

    # ── Use a real quantized MNIST sample for the MLIR trace ────────────
    try:
        from torchvision import datasets, transforms
        _tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        ds         = datasets.MNIST("./data", train=False, download=True, transform=_tf)
        img, label = ds[0]                                  # (1, 28, 28) float
        x_int8     = torch.clamp(
            torch.round(img.view(-1) / scales["input"]), -128, 127
        ).to(torch.int8)
        x_sample   = x_int8.tolist()
        print(f"Using MNIST test sample 0  (label = {label})")
    except Exception:
        x_sample = [0] * 784
        print("(torchvision unavailable – using zero input for MLIR trace)")

    generate_lenet_mlir(
        model=model,
        input_data=x_sample,
        batch_size=1,
        output_file=output_file,
        scales=scales,
    )


# ════════════════════════════════════════════════════════════════════════════
#  CLI entry point
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate MLIR for quantized LeNet-5."
    )
    parser.add_argument(
        "--lenet", metavar="JSON", required=True,
        help=(
            "Path to lenet_quantized.json produced by train.py. "
            "Generates lenet_mnist.mlir."
        ),
    )
    parser.add_argument(
        "--gemmini-pad-conv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pad LeNet conv int_mm dimensions to Gemmini-friendly multiples of 16 (default: enabled).",
    )
    parser.add_argument(
        "--gemmini-conv-chunk-rows",
        type=int,
        default=0,
        help="Split each conv int_mm into fixed row chunks (default: 0, disabled).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional MLIR output path (default: lenet_mnist.mlir or mlp_mnist_784x128x10.mlir).",
    )
    args = parser.parse_args()

    run_lenet_example(
        args.lenet,
        gemmini_pad_conv=args.gemmini_pad_conv,
        gemmini_chunk_rows=args.gemmini_conv_chunk_rows,
        output_file=(args.output if args.output else "lenet_mnist.mlir"),
    )


if __name__ == "__main__":
    main()
