#!/bin/bash
# =============================================================================
# torch-mlir from-source setup in conda env "nmj0"
# Based on: https://github.com/llvm/torch-mlir/blob/main/docs/development.md
# Tested on: Ubuntu 22.04 / 24.04
# =============================================================================
set -e  # exit on any error

# ─── 0. System dependencies ──────────────────────────────────────────────────
echo ">>> Installing system dependencies..."
sudo apt update
sudo apt install -y \
    python3-dev \
    cmake \
    ninja-build \
    clang \
    lld \
    ccache \
    git \
    wget \
    build-essential

# ─── 1. Create & activate conda env ──────────────────────────────────────────
echo ">>> Creating conda env 'nmj0' with Python 3.11..."
conda create -n nmj0 python=3.11 -y

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate nmj0

python -m pip install --upgrade pip

# ─── 2. Clone torch-mlir + submodules ────────────────────────────────────────
# Clone into current directory (no cd ~)
echo ">>> Cloning torch-mlir into $(pwd)/torch-mlir ..."
git clone https://github.com/NMJ0/torch-mlir.git
cd torch-mlir
TORCH_MLIR_DIR=$(pwd)   # save absolute path for later

echo ">>> Initializing submodules (this will take a while)..."
git submodule update --init --depth=1 --progress

# ─── 3. Install Python requirements ──────────────────────────────────────────
echo ">>> Installing Python requirements..."
pip install -r requirements.txt -r torchvision-requirements.txt

# ─── 4. CMake configure (in-tree build with fast compile flags) ───────────────
echo ">>> Configuring CMake..."
cmake -GNinja -Bbuild \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DPython3_FIND_VIRTUALENV=ONLY \
    -DPython_FIND_VIRTUALENV=ONLY \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DLLVM_TARGETS_TO_BUILD=host \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_EXTERNAL_PROJECTS="torch-mlir" \
    -DLLVM_EXTERNAL_TORCH_MLIR_SOURCE_DIR="$TORCH_MLIR_DIR" \
    -DTORCH_MLIR_ENABLE_PYTORCH_EXTENSIONS=ON \
    -DTORCH_MLIR_ENABLE_JIT_IR_IMPORTER=ON \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DLLVM_USE_LINKER=lld \
    externals/llvm-project/llvm

# ─── 5. Build ────────────────────────────────────────────────────────────────
echo ">>> Building (this will take a long time)..."
cmake --build build

# ─── 6. Set up PYTHONPATH ────────────────────────────────────────────────────
echo ">>> Setting up PYTHONPATH..."
cd "$TORCH_MLIR_DIR"
./build_tools/write_env_file.sh
source ./.env && export PYTHONPATH

# Persist PYTHONPATH into the conda env so it's set on every `conda activate`
CONDA_ENV_DIR="$CONDA_BASE/envs/nmj0"
mkdir -p "$CONDA_ENV_DIR/etc/conda/activate.d"
cat > "$CONDA_ENV_DIR/etc/conda/activate.d/torch_mlir_env.sh" << EOF
source $TORCH_MLIR_DIR/.env
export PYTHONPATH
EOF

# ─── 7. Verify ───────────────────────────────────────────────────────────────
echo ">>> Verifying installation..."
python -c "
import torch
import torchvision
import torch_mlir
print('torch      :', torch.__version__)
print('torchvision:', torchvision.__version__)
print('torch_mlir : OK')
"

echo ""
echo "✅ Done! To use this env in future sessions:"
echo "   conda activate nmj0"
echo ""
echo "To run the ResNet18 example:"
echo "   cd $TORCH_MLIR_DIR"
echo "   python projects/pt1/examples/fximporter_resnet18.py"
