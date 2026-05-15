"""
train.py  –  Train a 3-layer MLP on MNIST then post-training-quantize it.

Pipeline
--------
1.  Train a float32 MLP  (784 → 128 → 10)  with cross-entropy.
2.  Quantize weights symmetrically  (per-tensor, int8).
3.  Run a calibration pass to measure activation ranges.
4.  Derive all quantization scales:
        scale_input, scale_w1, scale_hidden, scale_w2
        scale_acc1  = scale_input  × scale_w1   (FC1 int32 accumulator)
        scale_acc2  = scale_hidden × scale_w2   (FC2 int32 accumulator)
        requant_fc1 = scale_acc1   / scale_hidden  (int32 → int8 after ReLU)
5.  Quantize biases into int32 using their accumulator scale.
6.  Verify accuracy of the fully-quantized integer model.
7.  Serialize everything to  mnist_quantized.json  for the inference script.

Quantization convention (symmetric, per-tensor)
------------------------------------------------
    scale  = max(|x|) / 127
    x_int8 = clamp( round(x_float / scale), -128, 127 )

Forward pass in integer arithmetic
-----------------------------------
    # Input quantization
    x_int8  = quant(x_float, scale_input)

    # FC1
    acc1    = x_int8 ⊗ w1_int8 + b1_int32        # int8×int8→int32
    acc1    = relu(acc1)
    h_int8  = requant(acc1, requant_fc1)           # int32→int8

    # FC2
    logits  = h_int8 ⊗ w2_int8 + b2_int32         # int8×int8→int32
    pred    = argmax(logits)
"""

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ════════════════════════════════════════════════════════════════════════════
#  Hyper-parameters
# ════════════════════════════════════════════════════════════════════════════

LAYER_SIZES   = [784, 128, 10]   # [in, hidden, out]
BATCH_SIZE    = 256
EPOCHS        = 5
LR            = 1e-3
CALIB_BATCHES = 20               # calibration mini-batches (≈5 120 images)
SAVE_PATH     = "mnist_quantized.json"

# ════════════════════════════════════════════════════════════════════════════
#  Data loaders
# ════════════════════════════════════════════════════════════════════════════

_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

train_ds = datasets.MNIST("./data", train=True,  download=True, transform=_transform)
test_ds  = datasets.MNIST("./data", train=False, download=True, transform=_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
calib_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ════════════════════════════════════════════════════════════════════════════
#  Float model
# ════════════════════════════════════════════════════════════════════════════

class FloatMLP(nn.Module):
    """Standard float32 MLP used for training."""

    def __init__(self, sizes: list[int]):
        super().__init__()
        self.fc1  = nn.Linear(sizes[0], sizes[1])
        self.relu = nn.ReLU()
        self.fc2  = nn.Linear(sizes[1], sizes[2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.fc2(self.relu(self.fc1(x)))


# ════════════════════════════════════════════════════════════════════════════
#  Training
# ════════════════════════════════════════════════════════════════════════════

def train(model: FloatMLP) -> None:
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    print("=" * 60)
    print("Training float model …")
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


def evaluate_float(model: FloatMLP) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            preds = model(xb).argmax(1)
            correct += (preds == yb).sum().item()
            total   += yb.size(0)
    return correct / total


# ════════════════════════════════════════════════════════════════════════════
#  Quantization helpers
# ════════════════════════════════════════════════════════════════════════════

def compute_scale(tensor: torch.Tensor) -> float:
    """Symmetric per-tensor scale: maps max(|x|) → 127."""
    return float(tensor.abs().max().item()) / 127.0


def quantize_to_int8(tensor: torch.Tensor, scale: float) -> torch.Tensor:
    """Float tensor → int8 tensor (symmetric, per-tensor)."""
    return torch.clamp(torch.round(tensor / scale), -128, 127).to(torch.int8)


# ════════════════════════════════════════════════════════════════════════════
#  Post-Training Quantization  (PTQ)
# ════════════════════════════════════════════════════════════════════════════

def ptq(model: FloatMLP) -> dict:
    """
    Run PTQ on the trained float model.

    Returns a dict with all int8 weights, int32 biases, and every
    quantization scale needed by the inference script.
    """
    model.eval()

    # ── 1. Weight quantization ──────────────────────────────────────────
    w1f = model.fc1.weight.data   # (128, 784)  float32
    b1f = model.fc1.bias.data     # (128,)      float32
    w2f = model.fc2.weight.data   # (10,  128)  float32
    b2f = model.fc2.bias.data     # (10,)       float32

    scale_w1 = compute_scale(w1f)
    scale_w2 = compute_scale(w2f)

    w1_int8 = quantize_to_int8(w1f, scale_w1)   # (128, 784) int8
    w2_int8 = quantize_to_int8(w2f, scale_w2)   # (10,  128) int8

    # ── 2. Activation calibration (running max |x|) ─────────────────────
    print("\nCalibrating activation scales …")
    max_input  = 0.0
    max_hidden = 0.0

    with torch.no_grad():
        for i, (xb, _) in enumerate(calib_loader):
            if i >= CALIB_BATCHES:
                break
            x_flat = xb.view(xb.size(0), -1)          # (B, 784)
            max_input  = max(max_input,  x_flat.abs().max().item())
            h_float    = model.relu(model.fc1(x_flat)) # (B, 128) float32
            max_hidden = max(max_hidden, h_float.abs().max().item())

    scale_input  = max_input  / 127.0
    scale_hidden = max_hidden / 127.0

    # ── 3. Accumulator scales ────────────────────────────────────────────
    #   acc1 (int32 after FC1) represents values at scale_input × scale_w1
    #   acc2 (int32 after FC2) represents values at scale_hidden × scale_w2
    scale_acc1 = scale_input  * scale_w1
    scale_acc2 = scale_hidden * scale_w2

    # ── 4. Requantization scale for FC1 output ───────────────────────────
    #   We want to convert acc1 (at scale_acc1) → int8 (at scale_hidden).
    #   int8_out = clamp( round( acc1_int32 × requant_fc1 ), -128, 127 )
    requant_fc1 = scale_acc1 / scale_hidden

    # ── 5. Bias quantization ─────────────────────────────────────────────
    #   bias must be in the same scale as the int32 accumulator so that
    #   acc_int32 = x_int8 ⊗ w_int8 + b_int32  is scale-consistent.
    #   b_int32 = round( b_float / scale_acc )
    b1_int32 = torch.round(b1f / scale_acc1).to(torch.int32)
    b2_int32 = torch.round(b2f / scale_acc2).to(torch.int32)

    # ── 6. Print scale summary ───────────────────────────────────────────
    print(f"  scale_input   = {scale_input:.8f}")
    print(f"  scale_w1      = {scale_w1:.8f}")
    print(f"  scale_acc1    = {scale_acc1:.8f}")
    print(f"  scale_hidden  = {scale_hidden:.8f}")
    print(f"  requant_fc1   = {requant_fc1:.8f}")
    print(f"  scale_w2      = {scale_w2:.8f}")
    print(f"  scale_acc2    = {scale_acc2:.8f}")

    return dict(
        w1       = w1_int8,
        b1       = b1_int32,
        w2       = w2_int8,
        b2       = b2_int32,
        scale_input  = scale_input,
        scale_w1     = scale_w1,
        scale_hidden = scale_hidden,
        scale_w2     = scale_w2,
        scale_acc1   = scale_acc1,
        scale_acc2   = scale_acc2,
        requant_fc1  = requant_fc1,
    )


# ════════════════════════════════════════════════════════════════════════════
#  Verify integer model on test set
# ════════════════════════════════════════════════════════════════════════════

def evaluate_quantized(q: dict) -> float:
    """
    Run the fully-integer forward pass and return test accuracy.
    Uses Python-level int32 arithmetic (no GPU needed).
    """
    w1   = q["w1"].to(torch.int32)
    b1   = q["b1"]
    w2   = q["w2"].to(torch.int32)
    b2   = q["b2"]
    scale_input  = q["scale_input"]
    requant_fc1  = q["requant_fc1"]

    correct = total = 0

    with torch.no_grad():
        for xb, yb in test_loader:
            x_flat = xb.view(xb.size(0), -1)

            # ── Quantize input ──────────────────────────────────────────
            x_int8 = quantize_to_int8(x_flat, scale_input)  # (B, 784) int8

            # ── FC1: int8 × int8 → int32 ────────────────────────────────
            acc1 = x_int8.to(torch.int32) @ w1.T + b1.unsqueeze(0)  # (B, 128) int32

            # ── ReLU on int32 ───────────────────────────────────────────
            acc1 = acc1.clamp(min=0)

            # ── Requantize → int8 ────────────────────────────────────────
            h_int8 = torch.clamp(
                torch.round(acc1.float() * requant_fc1), -128, 127
            ).to(torch.int8)                                 # (B, 128) int8

            # ── FC2: int8 × int8 → int32 ────────────────────────────────
            logits = h_int8.to(torch.int32) @ w2.T + b2.unsqueeze(0)  # (B, 10) int32

            preds   = logits.argmax(1)
            correct += (preds == yb).sum().item()
            total   += yb.size(0)

    return correct / total


# ════════════════════════════════════════════════════════════════════════════
#  Save to JSON
# ════════════════════════════════════════════════════════════════════════════

def save_quantized(q: dict, path: str) -> None:
    """
    Serialize the quantized model to JSON so mlp_inference.py can load it.

    Stored fields
    -------------
    layer_sizes    : [784, 128, 10]
    w1, b1, w2, b2 : int8 / int32 weight tensors as nested lists
    scales         : all float scales needed for requantization
    """
    payload = {
        "layer_sizes": LAYER_SIZES,
        "w1" : q["w1"].tolist(),   # (128, 784) int8
        "b1" : q["b1"].tolist(),   # (128,)     int32
        "w2" : q["w2"].tolist(),   # (10,  128) int8
        "b2" : q["b2"].tolist(),   # (10,)      int32
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
    print(f"\nSaved quantized model → '{path}'")


# ════════════════════════════════════════════════════════════════════════════
#  Entry point
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # ── 1. Train float model ─────────────────────────────────────────────
    float_model = FloatMLP(LAYER_SIZES)
    train(float_model)
    float_acc = evaluate_float(float_model)
    print(f"\nFloat model test accuracy : {float_acc * 100:.2f}%")

    # ── 2. Post-training quantization ────────────────────────────────────
    print("\n" + "=" * 60)
    print("Post-Training Quantization …")
    print("=" * 60)
    q = ptq(float_model)

    # ── 3. Verify quantized accuracy ─────────────────────────────────────
    print("\nVerifying quantized model on test set …")
    q_acc = evaluate_quantized(q)
    print(f"Quantized model test accuracy : {q_acc * 100:.2f}%")
    print(f"Accuracy drop from quantization : {(float_acc - q_acc) * 100:.2f}pp")

    # ── 4. Save ───────────────────────────────────────────────────────────
    save_quantized(q, SAVE_PATH)

    print("\nDone!  Run  python mlp_inference.py  to generate MLIR.")


if __name__ == "__main__":
    main()