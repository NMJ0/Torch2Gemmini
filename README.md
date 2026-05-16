# MLIR-Gemmini

<details>
<summary><strong>Steps to set up forked CHIPYARD</strong></summary>

<br>

- Clone the forked chipyard repository 

```bash
git clone https://github.com/NMJ0/chipyard.git
```

- Build chipyard ecosystem 

```bash
cd chipyard
./build-setup.sh riscv-tools
```

- If any error related to firtool comes -- Steps to install firtools  
(this is what worked for me, you can try to set up firtools in your own way)

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

</details>



<details>
<summary><strong>Steps to set up forked buddy-mlir</strong></summary>

<br>

- Install dependencies

```bash
sudo apt install flatbuffers-compiler libflatbuffers-dev libnuma-dev
```

- Clone and initialize the forked buddy-mlir repo

```bash
git clone https://github.com/NMJ0/buddy-mlir.git
cd buddy-mlir
git submodule update --init llvm
```

- pip install requirements

```bash
pip install -r requirements.txt
```

- Build and test LLVM

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

- Build BUDDY

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

</details>



<details>
<summary><strong>Steps to set up forked torch-mlir</strong></summary>

<br>

```bash
chmod +x torch_mlir_setup.sh
./torch_mlir_setup.sh
```

</details>

<details>
<summary><strong>FPGA Programming Guide</strong></summary>

<br>



## 1. Enable the Chipyard Environment

Navigate to the Chipyard directory and enable the environment.

```bash
(base) noel@sjd:~$ cd chipyard/
(base) noel@sjd:~/chipyard$ source env.sh 
(/home/noel/chipyard/.conda-env) noel@sjd:~/chipyard$
```

This activates the Chipyard Conda environment required for building and running tools.

## 2. Enable Vivado 2018 NFS Server

Mount the hardware tools and enable Vivado 2018.

```bash
(/home/noel/chipyard/.conda-env) noel@sjd:~/chipyard/fpga$ hwmountnfs
[sudo] password for noel: 
(/home/noel/chipyard/.conda-env) noel@sjd:~/chipyard/fpga$ v18
(/home/noel/chipyard/.conda-env) noel@sjd:~/chipyard/fpga$ which vivado 
/home/noel/tools/hw_tools/Vivado/Vivado/2018.3/bin/vivado
```

This ensures the correct Vivado version is active.

## 3. Generate Bitstream for Rocket + Gemmini 

Generate the FPGA bitstream using the configuration defined in Configs.scala.

```bash
(/home/noel/chipyard/.conda-env) noel@sjd:~/chipyard/fpga$ make SUB_PROJECT=nexysvideo CONFIG=RocketGemminiNexysVideo10MHzConfig bitstream
```

The generated bitstream will be placed in the generated-src directory.

## 4. Build ELF Binaries for C Programs

Navigate to the Gemmini software tests and build the binaries.

```bash 
(/home/noel/chipyard/.conda-env) noel@sjd:~/chipyard/generators/gemmini/software/gemmini-rocc-tests$ ./build.sh

```

All C programs are compiled into ELF binaries and stored in the build directory.

## 5. Connect and Program the  FPGA

Steps to program the FPGA:

1. Open Vivado Hardware Manager

2. Connect both:

- UART cable

- FPGA programming cable

3. Click Auto Connect

4. Click Program Device

5. Select the bitstream file from:

/home/noel/chipyard/fpga/generated-src

6. Program the device.

Important: Reset the CPU before running each program.(There is a CPU Reset button on the FPGA )

## 6. Identify the Connected USB Port

Check which UART port corresponds to the FPGA.

```bash 
(/home/noel/chipyard/.conda-env) noel@sjd:~/chipyard$ ls /dev/ttyUSB*
/dev/ttyUSB0
```

## 7. Running Programs on the FPGA

- Running Hello World on the Rocket Core

```bash 
(/home/noel/chipyard/.conda-env) noel@sjd:~/chipyard$ ~/chipyard/.conda-env/riscv-tools/bin/uart_tsi   +tty=/dev/ttyUSB0 +baudrate=115200   +selfcheck software/baremetal-ide/build/examples/chipyard-tests/hello.elf
```

- Running the Gemmini Template Test

This program multiplies a matrix with an identity matrix and verifies that the output matches the input.

```bash 
(/home/noel/chipyard/.conda-env) noel@sjd:~/chipyard$ ~/chipyard/.conda-env/riscv-tools/bin/uart_tsi   +tty=/dev/ttyUSB0 +baudrate=115200   +selfcheck generators/gemmini/software/gemmini-rocc-tests/build/bareMetalC/template-baremetal
```

- Running Matrix Addition on Rocket + Gemmini



```bash
(/home/noel/chipyard/.conda-env) noel@sjd:~/chipyard$ ~/chipyard/.conda-env/riscv-tools/bin/uart_tsi   +tty=/dev/ttyUSB0 +baudrate=115200   +selfcheck generators/gemmini/software/gemmini-rocc-tests/build/bareMetalC/matrix_add-baremetal
```

</details>
