import torch
from torch_mlir import torchscript


class MatMulModule(torch.nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # torch._int_mm: true int8 x int8 -> int32, no promotion
        return torch._int_mm(a, b)


def main():
    model = MatMulModule().eval()

    a = torch.tensor(
        [
            [21, 22, 23, 24, 25, 26, 27, 28],
            [29, 30, 31, 32, 33, 34, 35, 36],
            [37, 38, 39, 40, 41, 42, 43, 44],
            [45, 46, 47, 48, 49, 50, 51, 52],
            [53, 54, 55, 56, 57, 58, 59, 60],
            [61, 62, 63, 64, 65, 66, 67, 68],
            [69, 70, 71, 72, 73, 74, 75, 76],
            [77, 78, 79, 80, 81, 82, 83, 84],
        ],
        dtype=torch.int8,
    )

    b = torch.tensor(
        [
            [84, 83, 82, 81, 80, 79, 78, 77],
            [76, 75, 74, 73, 72, 71, 70, 69],
            [68, 67, 66, 65, 64, 63, 62, 61],
            [60, 59, 58, 57, 56, 55, 54, 53],
            [52, 51, 50, 49, 48, 47, 46, 45],
            [44, 43, 42, 41, 40, 39, 38, 37],
            [36, 35, 34, 33, 32, 31, 30, 29],
            [28, 27, 26, 25, 24, 23, 22, 21],
        ],
        dtype=torch.int8,
    )

    # Ground truth using _int_mm directly
    expected = torch._int_mm(a, b)

    mlir_module = torchscript.compile(
        model,
        (a, b),
        output_type=torchscript.OutputType.LINALG_ON_TENSORS,
    )
    mlir_text = str(mlir_module)

    with open("matmul_from_torch.mlir", "w") as f:
        f.write(mlir_text)

    print("Input matrix A (int8):")
    print(a)
    print("\nInput matrix B (int8):")
    print(b)
    print("\nExpected output A x B (int32):")
    print(expected)
    print("\nTorch-MLIR output written to matmul_from_torch.mlir\n")
    print(mlir_text)


if __name__ == "__main__":
    main()