# MLIR-Gemmini
## Set up CHIPYARD
- Clone the forked chipyard repository 
```bash
git clone https://github.com/NMJ0/chipyard.git
```
- build chipyard ecosystem 
```bash
cd chipyard
./build-setup.sh riscv-tools  
```
- if any error related to firtool comes --Steps  to install firtools (this is what worked for me , you can try to set up firtools in your own way)

```bash
# Step 1: Remove the broken firtool
rm /home/noel/chipyard/.conda-env/bin/firtool   ## change to your system's path

# Step 2: Download full CIRCT distribution (includes libraries)
cd /tmp
rm -rf circt-dist
mkdir circt-dist && cd circt-dist
wget -q -O - https://github.com/llvm/circt/releases/download/firtool-1.56.1/circt-full-shared-linux-x64.tar.gz | tar -zx

# Step 3: Copy entire directory to conda environment
cp -r firtool-1.56.1/* $CONDA_PREFIX/

# Step 4: Verify - check both binary and libraries exist
ls -la $CONDA_PREFIX/bin/firtool
ls -la $CONDA_PREFIX/lib/libCIRCT*.so

# Step 5: Test
firtool --version
```
## Set up buddy-mlir
- install dependencies
```bash
sudo apt install flatbuffers-compiler libflatbuffers-dev libnuma-dev
```

- clone and initialize the forked buddy-mlir repo

```bash
git clone https://github.com/NMJ0/buddy-mlir.git
cd buddy-mlir
git submodule update --init llvm
```
- pip install requirements
```bash
pip install -r requirements.txt
```
- build and test LLVM
```bash
 cd buddy-mlir
 mkdir llvm/build
 cd llvm/build
 cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="mlir;clang;openmp" \
    -DLLVM_TARGETS_TO_BUILD="host;RISCV" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DOPENMP_ENABLE_LIBOMPTARGET=OFF \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DPython3_EXECUTABLE=$(which python3)
 ninja check-clang check-mlir omp
```
- build BUDDY
```bash
 cd buddy-mlir
 mkdir build
 cd build
 cmake -G Ninja .. \
    -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir \
    -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON \
    -DPython3_EXECUTABLE=$(which python3)
 ninja
 ninja check-buddy
```
## set up torch-mlir
```bash
chmod +x torch_mlir_setup.sh
./torch_mlir_setup.sh
```

