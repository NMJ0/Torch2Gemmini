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

