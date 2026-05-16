"""
train.py  –  Train LeNet-5 on MNIST then post-training-quantize it.

Maximally reuses the MLP pipeline: compute_scale, quantize_to_int8,
train(), evaluate_float(), and the calibration/requant logic are all
identical in structure; only the model definition and the PTQ activation
collection differ because LeNet has conv layers.

LeNet-5 architecture (valid-padded, stride-1 convs)
----------------------------------------------------
    Conv1  (1→6,  5×5) → ReLU → MaxPool(2×2) → (N,6,12,12)
    Conv2  (6→16, 5×5) → ReLU → MaxPool(2×2) → (N,16,4,4)
    Flatten                                    → (N,256)
    FC1    (256→120)   → ReLU
    FC2    (120→84)    → ReLU
    FC3    (84→10)                             → int32 logits

Quantization data-flow (per layer, symmetric per-tensor)
---------------------------------------------------------
    scale_acc  = scale_input_to_layer × scale_weight
    requant    = scale_acc / scale_output_activation
    b_int32    = round(b_float / scale_acc)

Integer forward (verified in evaluate_lenet_quantized)
------------------------------------------------------
    x_int8  = quant(x_float, scale_input)
    # Conv1
    acc_c1  = conv_int8×int8→int32(x_int8, wc1) + bc1_int32
    acc_c1  = relu_int32(acc_c1)
    h_c1    = requant(acc_c1, requant_c1)        # int32 → int8
    h_c1    = max_pool_int8(h_c1)
    # Conv2
    acc_c2  = conv_int8×int8→int32(h_c1, wc2) + bc2_int32
    acc_c2  = relu_int32(acc_c2)
    h_c2    = requant(acc_c2, requant_c2)
    h_c2    = max_pool_int8(h_c2)
    # FC1
    acc_f1  = h_flat_int8 ⊗ wf1 + bf1_int32
    acc_f1  = relu_int32(acc_f1)
    h_f1    = requant(acc_f1, requant_f1)
    # FC2
    acc_f2  = h_f1 ⊗ wf2 + bf2_int32
    acc_f2  = relu_int32(acc_f2)
    h_f2    = requant(acc_f2, requant_f2)
    # FC3
    logits  = h_f2 ⊗ wf3 + bf3_int32            # int32, no activation
"""

import argparse
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ════════════════════════════════════════════════════════════════════════════
#  Hyper-parameters
# ════════════════════════════════════════════════════════════════════════════

# MLP sizes kept for backward compat (used by --mlp flag)
MLP_LAYER_SIZES = [784, 128, 10]

BATCH_SIZE    = 256
EPOCHS        = 10          # LeNet needs a few more epochs than the tiny MLP
LR            = 1e-3
CALIB_BATCHES = 20          # ≈5 120 images for calibration
LENET_SAVE_PATH = "lenet_quantized.json"
MLP_SAVE_PATH   = "mnist_quantized.json"

# ════════════════════════════════════════════════════════════════════════════
#  Data loaders  (shared by both MLP and LeNet)
# ════════════════════════════════════════════════════════════════════════════

_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

train_ds = datasets.MNIST("./data", train=True,  download=True, transform=_transform)
test_ds  = datasets.MNIST("./data", train=False, download=True, transform=_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
calib_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ════════════════════════════════════════════════════════════════════════════
#  Float models
# ════════════════════════════════════════════════════════════════════════════

class FloatMLP(nn.Module):
    """Standard float32 MLP used for training (kept for --mlp backward compat)."""
    def __init__(self, sizes: list[int]):
        super().__init__()
        self.fc1  = nn.Linear(sizes[0], sizes[1])
        self.relu = nn.ReLU()
        self.fc2  = nn.Linear(sizes[1], sizes[2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.fc2(self.relu(self.fc1(x)))


class FloatLeNet(nn.Module):
    """
    LeNet-5 for MNIST (float32).
    Input: (N, 1, 28, 28) — standard torchvision MNIST output shape.

    Architecture (valid convs, no padding)
    ---------------------------------------
        Conv(1→6, 5×5) → ReLU → MaxPool(2×2)   → (N, 6, 12, 12)
        Conv(6→16, 5×5) → ReLU → MaxPool(2×2)  → (N, 16, 4, 4)
        Flatten                                  → (N, 256)
        Linear(256→120) → ReLU
        Linear(120→84)  → ReLU
        Linear(84→10)                            → logits
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,  6,  5)   # valid: 28→24
        self.conv2 = nn.Conv2d(6,  16, 5)   # valid: 12→8
        self.fc1   = nn.Linear(256, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84,  10)
        self.relu  = nn.ReLU()
        self.pool  = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pool(self.relu(self.conv1(x)))   # (N, 6, 12, 12)
        h = self.pool(self.relu(self.conv2(h)))   # (N, 16, 4, 4)
        h = h.view(h.size(0), -1)                 # (N, 256)
        h = self.relu(self.fc1(h))                # (N, 120)
        h = self.relu(self.fc2(h))                # (N, 84)
        return self.fc3(h)                         # (N, 10)


# ════════════════════════════════════════════════════════════════════════════
#  Training  (unchanged from MLP version — works for any nn.Module)
# ════════════════════════════════════════════════════════════════════════════

def train(model: nn.Module) -> None:
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    print("=" * 60)
    print(f"Training {model.__class__.__name__} …")
    print("=" * 60)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        acc = evaluate_float(model)
        avg_loss = running_loss / len(train_loader)
        print(f"  Epoch {epoch}/{EPOCHS}  loss={avg_loss:.4f}  test_acc={acc*100:.2f}%")


def evaluate_float(model: nn.Module) -> float:
    """Works unchanged for both FloatMLP and FloatLeNet."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            preds = model(xb).argmax(1)
            correct += (preds == yb).sum().item()
            total   += yb.size(0)
    return correct / total


# ════════════════════════════════════════════════════════════════════════════
#  Quantization helpers  (unchanged — reused by both MLP and LeNet PTQ)
# ════════════════════════════════════════════════════════════════════════════

def compute_scale(tensor: torch.Tensor) -> float:
    """Symmetric per-tensor scale: maps max(|x|) → 127."""
    return float(tensor.abs().max().item()) / 127.0


def quantize_to_int8(tensor: torch.Tensor, scale: float) -> torch.Tensor:
    """Float tensor → int8 tensor (symmetric, per-tensor)."""
    return torch.clamp(torch.round(tensor / scale), -128, 127).to(torch.int8)


# ════════════════════════════════════════════════════════════════════════════
#  PTQ for the MLP  (kept for --mlp backward compat)
# ════════════════════════════════════════════════════════════════════════════

def ptq(model: FloatMLP) -> dict:
    """Post-training quantization for the 2-layer MLP (unchanged)."""
    model.eval()

    w1f = model.fc1.weight.data
    b1f = model.fc1.bias.data
    w2f = model.fc2.weight.data
    b2f = model.fc2.bias.data

    scale_w1 = compute_scale(w1f)
    scale_w2 = compute_scale(w2f)
    w1_int8  = quantize_to_int8(w1f, scale_w1)
    w2_int8  = quantize_to_int8(w2f, scale_w2)

    print("\nCalibrating MLP activation scales …")
    max_input = max_hidden = 0.0

    with torch.no_grad():
        for i, (xb, _) in enumerate(calib_loader):
            if i >= CALIB_BATCHES:
                break
            x_flat     = xb.view(xb.size(0), -1)
            max_input  = max(max_input,  x_flat.abs().max().item())
            h_float    = model.relu(model.fc1(x_flat))
            max_hidden = max(max_hidden, h_float.abs().max().item())

    scale_input  = max_input  / 127.0
    scale_hidden = max_hidden / 127.0
    scale_acc1   = scale_input  * scale_w1
    scale_acc2   = scale_hidden * scale_w2
    requant_fc1  = scale_acc1 / scale_hidden

    b1_int32 = torch.round(b1f / scale_acc1).to(torch.int32)
    b2_int32 = torch.round(b2f / scale_acc2).to(torch.int32)

    print(f"  scale_input   = {scale_input:.8f}")
    print(f"  scale_w1      = {scale_w1:.8f}")
    print(f"  scale_acc1    = {scale_acc1:.8f}")
    print(f"  scale_hidden  = {scale_hidden:.8f}")
    print(f"  requant_fc1   = {requant_fc1:.8f}")
    print(f"  scale_w2      = {scale_w2:.8f}")
    print(f"  scale_acc2    = {scale_acc2:.8f}")

    return dict(
        w1=w1_int8, b1=b1_int32, w2=w2_int8, b2=b2_int32,
        scale_input=scale_input, scale_w1=scale_w1,
        scale_hidden=scale_hidden, scale_w2=scale_w2,
        scale_acc1=scale_acc1, scale_acc2=scale_acc2,
        requant_fc1=requant_fc1,
    )


# ════════════════════════════════════════════════════════════════════════════
#  PTQ for LeNet  (new — mirrors ptq() but for 5 layers)
# ════════════════════════════════════════════════════════════════════════════

def lenet_ptq(model: FloatLeNet) -> dict:
    """
    Post-training quantization for LeNet-5.

    Reuses compute_scale() and quantize_to_int8() (same helpers as MLP PTQ).
    Calibration visits each activation tensor in the quantized forward order:
        input → conv1 pool output → conv2 pool output → fc1 → fc2
    """
    model.eval()

    # ── 1. Weight quantization ──────────────────────────────────────────
    wc1f = model.conv1.weight.data   # (6,  1,  5, 5)
    bc1f = model.conv1.bias.data     # (6,)
    wc2f = model.conv2.weight.data   # (16, 6,  5, 5)
    bc2f = model.conv2.bias.data     # (16,)
    wf1f = model.fc1.weight.data     # (120, 256)
    bf1f = model.fc1.bias.data       # (120,)
    wf2f = model.fc2.weight.data     # (84, 120)
    bf2f = model.fc2.bias.data       # (84,)
    wf3f = model.fc3.weight.data     # (10, 84)
    bf3f = model.fc3.bias.data       # (10,)

    # Per-tensor symmetric weight scales (reuse compute_scale)
    scale_wc1 = compute_scale(wc1f)
    scale_wc2 = compute_scale(wc2f)
    scale_wf1 = compute_scale(wf1f)
    scale_wf2 = compute_scale(wf2f)
    scale_wf3 = compute_scale(wf3f)

    # Quantize weights (reuse quantize_to_int8)
    wc1_int8 = quantize_to_int8(wc1f, scale_wc1)   # (6,  1,  5, 5)
    wc2_int8 = quantize_to_int8(wc2f, scale_wc2)   # (16, 6,  5, 5)
    wf1_int8 = quantize_to_int8(wf1f, scale_wf1)   # (120, 256)
    wf2_int8 = quantize_to_int8(wf2f, scale_wf2)   # (84, 120)
    wf3_int8 = quantize_to_int8(wf3f, scale_wf3)   # (10, 84)

    # ── 2. Activation calibration (running max |x|) ─────────────────────
    print("\nCalibrating LeNet activation scales …")
    max_input = max_c1 = max_c2 = max_f1 = max_f2 = 0.0

    with torch.no_grad():
        for i, (xb, _) in enumerate(calib_loader):
            if i >= CALIB_BATCHES:
                break
            # xb: (N, 1, 28, 28)
            x_flat   = xb.view(xb.size(0), -1)
            max_input = max(max_input, x_flat.abs().max().item())

            # Conv1 path
            h = model.relu(model.conv1(xb))        # (N, 6, 24, 24)
            h = model.pool(h)                       # (N, 6, 12, 12)
            max_c1 = max(max_c1, h.abs().max().item())

            # Conv2 path
            h = model.relu(model.conv2(h))          # (N, 16, 8, 8)
            h = model.pool(h)                       # (N, 16, 4, 4)
            max_c2 = max(max_c2, h.abs().max().item())

            # FC1
            h = h.view(h.size(0), -1)              # (N, 256)
            h = model.relu(model.fc1(h))            # (N, 120)
            max_f1 = max(max_f1, h.abs().max().item())

            # FC2
            h = model.relu(model.fc2(h))            # (N, 84)
            max_f2 = max(max_f2, h.abs().max().item())

    scale_input = max_input / 127.0
    scale_c1    = max_c1    / 127.0
    scale_c2    = max_c2    / 127.0
    scale_f1    = max_f1    / 127.0
    scale_f2    = max_f2    / 127.0

    # ── 3. Accumulator scales (same formula as MLP: s_in × s_w) ────────
    scale_acc_c1 = scale_input * scale_wc1
    scale_acc_c2 = scale_c1   * scale_wc2
    scale_acc_f1 = scale_c2   * scale_wf1
    scale_acc_f2 = scale_f1   * scale_wf2
    scale_acc_f3 = scale_f2   * scale_wf3

    # ── 4. Requantization scales (int32 → int8 after each ReLU layer) ───
    requant_c1 = scale_acc_c1 / scale_c1
    requant_c2 = scale_acc_c2 / scale_c2
    requant_f1 = scale_acc_f1 / scale_f1
    requant_f2 = scale_acc_f2 / scale_f2
    # FC3 produces final int32 logits → no requantization

    # ── 5. Bias quantization (same formula as MLP: b_float / scale_acc) ─
    bc1_int32 = torch.round(bc1f / scale_acc_c1).to(torch.int32)
    bc2_int32 = torch.round(bc2f / scale_acc_c2).to(torch.int32)
    bf1_int32 = torch.round(bf1f / scale_acc_f1).to(torch.int32)
    bf2_int32 = torch.round(bf2f / scale_acc_f2).to(torch.int32)
    bf3_int32 = torch.round(bf3f / scale_acc_f3).to(torch.int32)

    # ── 6. Scale summary ────────────────────────────────────────────────
    print(f"  scale_input  = {scale_input:.8f}")
    print(f"  scale_wc1    = {scale_wc1:.8f}   scale_acc_c1 = {scale_acc_c1:.8f}   requant_c1 = {requant_c1:.8f}")
    print(f"  scale_c1     = {scale_c1:.8f}")
    print(f"  scale_wc2    = {scale_wc2:.8f}   scale_acc_c2 = {scale_acc_c2:.8f}   requant_c2 = {requant_c2:.8f}")
    print(f"  scale_c2     = {scale_c2:.8f}")
    print(f"  scale_wf1    = {scale_wf1:.8f}   scale_acc_f1 = {scale_acc_f1:.8f}   requant_f1 = {requant_f1:.8f}")
    print(f"  scale_f1     = {scale_f1:.8f}")
    print(f"  scale_wf2    = {scale_wf2:.8f}   scale_acc_f2 = {scale_acc_f2:.8f}   requant_f2 = {requant_f2:.8f}")
    print(f"  scale_f2     = {scale_f2:.8f}")
    print(f"  scale_wf3    = {scale_wf3:.8f}   scale_acc_f3 = {scale_acc_f3:.8f}")

    return dict(
        wc1=wc1_int8, bc1=bc1_int32,
        wc2=wc2_int8, bc2=bc2_int32,
        wf1=wf1_int8, bf1=bf1_int32,
        wf2=wf2_int8, bf2=bf2_int32,
        wf3=wf3_int8, bf3=bf3_int32,
        scale_input =scale_input,
        scale_wc1=scale_wc1, scale_wc2=scale_wc2,
        scale_wf1=scale_wf1, scale_wf2=scale_wf2, scale_wf3=scale_wf3,
        scale_c1=scale_c1,   scale_c2=scale_c2,
        scale_f1=scale_f1,   scale_f2=scale_f2,
        scale_acc_c1=scale_acc_c1, scale_acc_c2=scale_acc_c2,
        scale_acc_f1=scale_acc_f1, scale_acc_f2=scale_acc_f2,
        scale_acc_f3=scale_acc_f3,
        requant_c1=requant_c1, requant_c2=requant_c2,
        requant_f1=requant_f1, requant_f2=requant_f2,
    )


# ════════════════════════════════════════════════════════════════════════════
#  Integer forward verification  (mirrors evaluate_quantized for MLP)
# ════════════════════════════════════════════════════════════════════════════

def requant_int32_to_int8(acc: torch.Tensor, scale: float) -> torch.Tensor:
    """Shared requantization helper (same formula as RequantizeToInt8 in generate_mlir.py)."""
    return torch.clamp(
        torch.round(acc.float() * scale), -128, 127
    ).to(torch.int8)


def evaluate_lenet_quantized(q: dict) -> float:
    """
    Run the fully-integer LeNet forward and return test accuracy.

    Convolutions are simulated with F.conv2d on float32, which gives
    exactly the same results as im2col + int_mm because int8 values are
    exactly representable in float32 and convolution is linear.
    """
    def as_tensor(v, dtype):
        return v.to(dtype) if isinstance(v, torch.Tensor) else torch.tensor(v, dtype=dtype)

    # Reconstruct int tensors (cast weights to int32 for accumulators)
    wc1 = as_tensor(q["wc1"], torch.int8).reshape(6,  1,  5, 5).float()
    wc2 = as_tensor(q["wc2"], torch.int8).reshape(16, 6,  5, 5).float()
    wf1 = as_tensor(q["wf1"], torch.int8).reshape(120, 256).to(torch.int32)
    wf2 = as_tensor(q["wf2"], torch.int8).reshape(84,  120).to(torch.int32)
    wf3 = as_tensor(q["wf3"], torch.int8).reshape(10,   84).to(torch.int32)

    bc1 = as_tensor(q["bc1"], torch.int32)   # (6,)
    bc2 = as_tensor(q["bc2"], torch.int32)   # (16,)
    bf1 = as_tensor(q["bf1"], torch.int32)   # (120,)
    bf2 = as_tensor(q["bf2"], torch.int32)   # (84,)
    bf3 = as_tensor(q["bf3"], torch.int32)   # (10,)

    # Support both in-memory PTQ dict (flat scale keys) and serialized JSON shape.
    if "scales" in q:
        scales      = q["scales"]
        scale_input = scales["input"]
        requant_c1  = scales["requant_c1"]
        requant_c2  = scales["requant_c2"]
        requant_f1  = scales["requant_f1"]
        requant_f2  = scales["requant_f2"]
    else:
        scale_input = q["scale_input"]
        requant_c1  = q["requant_c1"]
        requant_c2  = q["requant_c2"]
        requant_f1  = q["requant_f1"]
        requant_f2  = q["requant_f2"]

    correct = total = 0

    with torch.no_grad():
        for xb, yb in test_loader:
            N = xb.shape[0]

            # ── Quantize input ──────────────────────────────────────────
            # xb: (N, 1, 28, 28) float
            x_flat = xb.view(N, -1)
            x_int8 = quantize_to_int8(x_flat, scale_input)    # (N, 784) int8
            x_img  = x_int8.view(N, 1, 28, 28)                # (N, 1, 28, 28) int8

            # ── Conv1: int8×int8 → int32 (via float conv, exact) ──────
            acc_c1 = F.conv2d(x_img.float(), wc1).to(torch.int32)   # (N, 6, 24, 24)
            acc_c1 = acc_c1 + bc1.view(1, -1, 1, 1)
            acc_c1 = acc_c1.clamp(min=0)                             # ReLU int32
            h_c1   = requant_int32_to_int8(acc_c1, requant_c1)      # (N, 6, 24, 24) int8
            h_c1   = F.max_pool2d(h_c1.float(), 2, 2).to(torch.int8)  # (N, 6, 12, 12)

            # ── Conv2 ────────────────────────────────────────────────────
            acc_c2 = F.conv2d(h_c1.float(), wc2).to(torch.int32)   # (N, 16, 8, 8)
            acc_c2 = acc_c2 + bc2.view(1, -1, 1, 1)
            acc_c2 = acc_c2.clamp(min=0)
            h_c2   = requant_int32_to_int8(acc_c2, requant_c2)     # (N, 16, 8, 8) int8
            h_c2   = F.max_pool2d(h_c2.float(), 2, 2).to(torch.int8)  # (N, 16, 4, 4)

            # ── Flatten ────────────────────────────────────────────────
            h_flat = h_c2.view(N, 256)                              # (N, 256) int8

            # ── FC1: int8×int8 → int32 (same as MLP evaluate_quantized)
            acc_f1 = h_flat.to(torch.int32) @ wf1.T + bf1.unsqueeze(0)  # (N, 120)
            acc_f1 = acc_f1.clamp(min=0)
            h_f1   = requant_int32_to_int8(acc_f1, requant_f1)     # (N, 120) int8

            # ── FC2 ───────────────────────────────────────────────────
            acc_f2 = h_f1.to(torch.int32) @ wf2.T + bf2.unsqueeze(0)   # (N, 84)
            acc_f2 = acc_f2.clamp(min=0)
            h_f2   = requant_int32_to_int8(acc_f2, requant_f2)     # (N, 84) int8

            # ── FC3: int32 logits ────────────────────────────────────
            logits = h_f2.to(torch.int32) @ wf3.T + bf3.unsqueeze(0)   # (N, 10)

            preds   = logits.argmax(1)
            correct += (preds == yb).sum().item()
            total   += yb.size(0)

    return correct / total


# ════════════════════════════════════════════════════════════════════════════
#  Serialize quantized LeNet to JSON
# ════════════════════════════════════════════════════════════════════════════

def save_lenet_quantized(q: dict, path: str) -> None:
    """
    Mirror of save_quantized() for the MLP, extended for 5 layers.

    All int8 weight tensors are flattened (row-major) to lists.
    Shape metadata is stored separately so generate_mlir.py can reconstruct.
    """
    payload = {
        # ── Conv weights: (C_out, C_in, kH, kW) stored flat ────────────
        "wc1": q["wc1"].tolist(),  # (6,  1, 5, 5) → 150 int8 values
        "bc1": q["bc1"].tolist(),  # (6,)           → 6   int32 values
        "wc2": q["wc2"].tolist(),  # (16, 6, 5, 5)  → 2400 int8 values
        "bc2": q["bc2"].tolist(),  # (16,)
        # ── FC weights: (out, in) stored flat ───────────────────────────
        "wf1": q["wf1"].tolist(),  # (120, 256)
        "bf1": q["bf1"].tolist(),  # (120,)
        "wf2": q["wf2"].tolist(),  # (84,  120)
        "bf2": q["bf2"].tolist(),  # (84,)
        "wf3": q["wf3"].tolist(),  # (10,  84)
        "bf3": q["bf3"].tolist(),  # (10,)
        # ── All quantization scales ─────────────────────────────────────
        "scales": {
            "input"     : q["scale_input"],
            "wc1"       : q["scale_wc1"],
            "wc2"       : q["scale_wc2"],
            "wf1"       : q["scale_wf1"],
            "wf2"       : q["scale_wf2"],
            "wf3"       : q["scale_wf3"],
            "c1"        : q["scale_c1"],
            "c2"        : q["scale_c2"],
            "f1"        : q["scale_f1"],
            "f2"        : q["scale_f2"],
            "acc_c1"    : q["scale_acc_c1"],
            "acc_c2"    : q["scale_acc_c2"],
            "acc_f1"    : q["scale_acc_f1"],
            "acc_f2"    : q["scale_acc_f2"],
            "acc_f3"    : q["scale_acc_f3"],
            "requant_c1": q["requant_c1"],
            "requant_c2": q["requant_c2"],
            "requant_f1": q["requant_f1"],
            "requant_f2": q["requant_f2"],
        },
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved quantized LeNet → '{path}'")


# ════════════════════════════════════════════════════════════════════════════
#  MLP save (kept for --mlp backward compat)
# ════════════════════════════════════════════════════════════════════════════

def save_quantized(q: dict, path: str) -> None:
    payload = {
        "layer_sizes": MLP_LAYER_SIZES,
        "w1": q["w1"].tolist(), "b1": q["b1"].tolist(),
        "w2": q["w2"].tolist(), "b2": q["b2"].tolist(),
        "scales": {
            "input"      : q["scale_input"],
            "w1"         : q["scale_w1"],
            "hidden"     : q["scale_hidden"],
            "w2"         : q["scale_w2"],
            "acc1"       : q["scale_acc1"],
            "acc2"       : q["scale_acc2"],
            "requant_fc1": q["requant_fc1"],
        },
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved quantized MLP → '{path}'")


# ════════════════════════════════════════════════════════════════════════════
#  Export input sample binary  (for input_data.h regeneration)
# ════════════════════════════════════════════════════════════════════════════

def export_input_bin(scale_input: float, bin_path: str = "input.bin") -> None:
    """
    Quantize the first MNIST test image to int8 and save as a raw binary.

    After running this, regenerate input_data.h with:
        xxd -i input.bin > input_data.h
    Then rename the array/length symbols to match wrapper.c:
        sed -i 's/input_bin/input_bin/g' input_data.h  (already correct)

    The binary is 784 bytes (28×28), matching the wrapper's IN_FEATURES=784.
    """
    ds    = datasets.MNIST("./data", train=False, download=True,
                           transform=_transform)
    img, label = ds[0]                                  # (1, 28, 28) float
    img_flat   = img.view(-1)                           # (784,) float
    x_int8     = quantize_to_int8(img_flat, scale_input)   # (784,) int8

    raw = x_int8.numpy().astype("int8").tobytes()
    with open(bin_path, "wb") as f:
        f.write(raw)

    print(f"\nExported test sample 0 (label={label}) → '{bin_path}'")
    print(f"  Regenerate input_data.h:  xxd -i {bin_path} > input_data.h")
    print(f"  (rename array/len to 'input_bin' / 'input_bin_len' if xxd changed them)")


# ════════════════════════════════════════════════════════════════════════════
#  Entry points
# ════════════════════════════════════════════════════════════════════════════

def run_lenet() -> None:
    # 1. Train float LeNet
    float_model = FloatLeNet()
    train(float_model)
    float_acc = evaluate_float(float_model)
    print(f"\nFloat LeNet test accuracy : {float_acc * 100:.2f}%")

    # 2. Post-training quantization
    print("\n" + "=" * 60)
    print("Post-Training Quantization (LeNet) …")
    print("=" * 60)
    q = lenet_ptq(float_model)

    # 3. Verify quantized accuracy
    print("\nVerifying quantized LeNet on test set …")
    q_acc = evaluate_lenet_quantized(q)
    print(f"Quantized LeNet test accuracy  : {q_acc * 100:.2f}%")
    print(f"Accuracy drop from quantization: {(float_acc - q_acc) * 100:.2f} pp")

    # 4. Save weights + scales
    save_lenet_quantized(q, LENET_SAVE_PATH)

    # 5. Export input binary (so Makefile can regenerate input_data.h)
    export_input_bin(q["scale_input"], "input.bin")

    print("\nDone! Run  python generate_mlir.py --lenet lenet_quantized.json")


def run_mlp() -> None:
    """Unchanged MLP pipeline (kept for backward compat with --mlp flag)."""
    float_model = FloatMLP(MLP_LAYER_SIZES)
    train(float_model)
    float_acc = evaluate_float(float_model)
    print(f"\nFloat MLP test accuracy : {float_acc * 100:.2f}%")

    print("\n" + "=" * 60)
    print("Post-Training Quantization (MLP) …")
    print("=" * 60)
    q = ptq(float_model)
    save_quantized(q, MLP_SAVE_PATH)
    print(f"\nDone! Run  python generate_mlir.py --mnist {MLP_SAVE_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and quantize LeNet-5 (or the original MLP) on MNIST."
    )
    parser.add_argument(
        "--mlp", action="store_true",
        help="Train the 784→128→10 MLP instead of LeNet-5 (backward compat).",
    )
    args = parser.parse_args()

    if args.mlp:
        run_mlp()
    else:
        run_lenet()


if __name__ == "__main__":
    main()
