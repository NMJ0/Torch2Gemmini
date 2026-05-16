SHELL := /bin/bash
ROOT_DIR ?= $(abspath ..)
CONDA ?= conda
CONDA_ENV ?= nmj0
TORCH_MLIR_PYTHON ?= python3
BUDDY_MLIR_DIR ?= $(ROOT_DIR)/buddy-mlir
BUDDY_OPT ?= $(BUDDY_MLIR_DIR)/build/bin/buddy-opt
BUDDY_TRANSLATE ?= $(BUDDY_MLIR_DIR)/build/bin/buddy-translate
BUDDY_LLC ?= $(BUDDY_MLIR_DIR)/build/bin/buddy-llc
RISCV_GCC ?= riscv64-unknown-elf-gcc
CHIPYARD_DIR ?= $(ROOT_DIR)/chipyard
CHIPYARD_GCC ?= $(CHIPYARD_DIR)/.conda-env/riscv-tools/bin/riscv64-unknown-elf-gcc
SPIKE ?= spike
UART_TSI ?= $(CHIPYARD_DIR)/.conda-env/riscv-tools/bin/uart_tsi
TTY ?= /dev/ttyUSB0
BAUDRATE ?= 115200
CHIPYARD_BAREMETAL ?= $(CHIPYARD_DIR)/software/baremetal-ide
CHIPYARD_BUILD ?= $(CHIPYARD_BAREMETAL)/build
FPGA_HEAP_SIZE ?= 0x80000
FPGA_STACK_SIZE ?= 0x10000

LENET_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
PY_SCRIPT := $(LENET_DIR)/generate_mlir.py
LENET_JSON := $(LENET_DIR)/lenet_quantized.json
TORCH_MLIR_FILE := $(LENET_DIR)/lenet_mnist.mlir

# Object files for the two lowering strategies
OBJ_GEMMINI := lenet_log_gemmini.o
OBJ_STD     := lenet_log_std.o

WRAPPER := $(LENET_DIR)/wrapper.c
EXE_GEMMINI := lenet_gemmini.elf
EXE_STD     := lenet_std.elf
FPGA_EXE_GEMMINI := lenet-fpga_gemmini.elf
FPGA_EXE_STD     := lenet-fpga_std.elf

# Python path for torch_mlir
TORCH_MLIR_SRC ?= $(ROOT_DIR)/torch-mlir
TORCH_MLIR_BUILD ?= $(TORCH_MLIR_SRC)/build
PYTHONPATH_TORCH_MLIR := $(TORCH_MLIR_SRC)/projects/pt1/python:$(TORCH_MLIR_SRC)/python:$(TORCH_MLIR_BUILD)/tools/torch-mlir/python_packages/torch_mlir

.PHONY: all all-std gen_mlir lower-gemmini lower-std link-gemmini link-std \
	run-gemmini run-std run-uart-gemmini run-uart-std clean

# Default target: Gemmini version (as before)
all: $(EXE_GEMMINI)

# New target: standard (no Gemmini) version
all-std: $(EXE_STD)

# -----------------------------------------------------------------------------
# MLIR generation (uses a real quantized image as trace)
# -----------------------------------------------------------------------------
gen_mlir: $(TORCH_MLIR_FILE)
	@echo "Generated MLIR: $(TORCH_MLIR_FILE)"

$(TORCH_MLIR_FILE): $(PY_SCRIPT) $(LENET_JSON)
	$(CONDA) run -n $(CONDA_ENV) env PYTHONPATH=$(PYTHONPATH_TORCH_MLIR) $(TORCH_MLIR_PYTHON) $(PY_SCRIPT) --lenet $(LENET_JSON) --gemmini-pad-conv --gemmini-conv-chunk-rows 0 --output $(TORCH_MLIR_FILE)

# -----------------------------------------------------------------------------
# Lowering to Gemmini (original flow)
# -----------------------------------------------------------------------------
lower-gemmini: $(OBJ_GEMMINI)

$(OBJ_GEMMINI): $(TORCH_MLIR_FILE)
	@echo "=== Gemmini lowering ==="
	$(BUDDY_OPT) $(TORCH_MLIR_FILE) \
		-one-shot-bufferize='bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map' \
		-o step1_bufferized.mlir
	$(BUDDY_OPT) step1_bufferized.mlir \
		-convert-bufferization-to-memref \
		-o step2_memref.mlir
	$(BUDDY_OPT) step2_memref.mlir \
		-convert-linalg-to-gemmini \
		-o step3_gemmini.mlir
	$(BUDDY_OPT) step3_gemmini.mlir \
		-expand-strided-metadata \
		-o step4_metadata.mlir
	$(BUDDY_OPT) step4_metadata.mlir \
		-convert-linalg-to-loops \
		-o step5_loops.mlir
	$(BUDDY_OPT) step5_loops.mlir \
		-lower-gemmini \
		-convert-math-to-llvm \
		-o step6_lowered.mlir
	$(BUDDY_TRANSLATE) step6_lowered.mlir \
		-buddy-to-llvmir \
		-o step7_llvm.ll
	$(BUDDY_LLC) -filetype=obj -mtriple=riscv64 -mattr=+buddyext,+D -float-abi=hard \
	    -relocation-model=pic \
		step7_llvm.ll -o $(OBJ_GEMMINI)

# -----------------------------------------------------------------------------
# Lowering to standard RISC‑V (no Gemmini)
# -----------------------------------------------------------------------------
lower-std: $(OBJ_STD)

$(OBJ_STD): $(TORCH_MLIR_FILE)
	@echo "=== Gemmini lowering ==="
	$(BUDDY_OPT) $(TORCH_MLIR_FILE) \
		-one-shot-bufferize='bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map' \
		-o step1_bufferized.mlir
	$(BUDDY_OPT) step1_bufferized.mlir \
		-convert-bufferization-to-memref \
		-o step2_memref.mlir
#	$(BUDDY_OPT) step2_memref.mlir \
		-convert-linalg-to-gemmini \
		-o step3_gemmini.mlir
	$(BUDDY_OPT) step2_memref.mlir \
		-expand-strided-metadata \
		-o step4_metadata.mlir
	$(BUDDY_OPT) step4_metadata.mlir \
		-convert-linalg-to-loops \
		-o step5_loops.mlir
	$(BUDDY_OPT) step5_loops.mlir \
		-lower-gemmini \
		-convert-math-to-llvm \
		-o step6_lowered.mlir
	$(BUDDY_TRANSLATE) step6_lowered.mlir \
		-buddy-to-llvmir \
		-o step7_llvm.ll
	$(BUDDY_LLC) -filetype=obj -mtriple=riscv64 -mattr=+buddyext,+D -float-abi=hard \
	    -relocation-model=pic \
		step7_llvm.ll -o $(OBJ_STD)

# -----------------------------------------------------------------------------
# Linking (Gemmini)
# -----------------------------------------------------------------------------
link-gemmini: $(EXE_GEMMINI)

$(EXE_GEMMINI): $(WRAPPER) $(OBJ_GEMMINI)
	$(RISCV_GCC) $(WRAPPER) $(OBJ_GEMMINI) -O2 -static -o $(EXE_GEMMINI)

# -----------------------------------------------------------------------------
# Linking (standard)
# -----------------------------------------------------------------------------
link-std: $(EXE_STD)

$(EXE_STD): $(WRAPPER) $(OBJ_STD)
	$(RISCV_GCC) $(WRAPPER) $(OBJ_STD) -O2 -static -o $(EXE_STD)

# -----------------------------------------------------------------------------
# Run on Spike (Gemmini)
# -----------------------------------------------------------------------------
run-gemmini: $(EXE_GEMMINI)
	$(SPIKE) --extension=gemmini pk $(EXE_GEMMINI)

# Run on Spike (standard)
run-std: $(EXE_STD)
	$(SPIKE)  pk $(EXE_STD)

# -----------------------------------------------------------------------------
# FPGA / bare‑metal linking (Gemmini)
# -----------------------------------------------------------------------------
link-fpga-gemmini: $(FPGA_EXE_GEMMINI)

$(FPGA_EXE_GEMMINI): $(WRAPPER) $(OBJ_GEMMINI)
	$(CHIPYARD_GCC) -g -static -nostartfiles -march=rv64imafd -mabi=lp64d -mcmodel=medany -specs=nosys.specs \
	  -fPIC \
	  -Wl,--defsym=__heap_size=$(FPGA_HEAP_SIZE) \
	  -Wl,--defsym=__stack_size=$(FPGA_STACK_SIZE) \
	  -T $(CHIPYARD_BAREMETAL)/glossy/glossy.ld \
	  $(WRAPPER) $(OBJ_GEMMINI) -o $(FPGA_EXE_GEMMINI) \
	  -L$(CHIPYARD_BUILD)/glossy \
	  -Wl,--whole-archive $(CHIPYARD_BUILD)/glossy/libglossy.a -Wl,--no-whole-archive \
	  $(CHIPYARD_BUILD)/platform/default/libchip-config.a \
	  $(CHIPYARD_BUILD)/driver/rocket-chip/rocketcore/librocketcore.a \
	  $(CHIPYARD_BUILD)/driver/rocket-chip/clint/libclint.a \
	  $(CHIPYARD_BUILD)/driver/rocket-chip/plic/libplic.a \
	  $(CHIPYARD_BUILD)/driver/htif/libhtif.a

# FPGA / bare‑metal linking (standard)
link-fpga-std: $(FPGA_EXE_STD)

$(FPGA_EXE_STD): $(WRAPPER) $(OBJ_STD)
	$(CHIPYARD_GCC) -g -static -nostartfiles -march=rv64imafd -mabi=lp64d -mcmodel=medany -specs=nosys.specs \
	  -fPIC \
	  -Wl,--defsym=__heap_size=$(FPGA_HEAP_SIZE) \
	  -Wl,--defsym=__stack_size=$(FPGA_STACK_SIZE) \
	  -T $(CHIPYARD_BAREMETAL)/glossy/glossy.ld \
	  $(WRAPPER) $(OBJ_STD) -o $(FPGA_EXE_STD) \
	  -L$(CHIPYARD_BUILD)/glossy \
	  -Wl,--whole-archive $(CHIPYARD_BUILD)/glossy/libglossy.a -Wl,--no-whole-archive \
	  $(CHIPYARD_BUILD)/platform/default/libchip-config.a \
	  $(CHIPYARD_BUILD)/driver/rocket-chip/rocketcore/librocketcore.a \
	  $(CHIPYARD_BUILD)/driver/rocket-chip/clint/libclint.a \
	  $(CHIPYARD_BUILD)/driver/rocket-chip/plic/libplic.a \
	  $(CHIPYARD_BUILD)/driver/htif/libhtif.a

# -----------------------------------------------------------------------------
# UART run (Gemmini)
# -----------------------------------------------------------------------------
run-uart-gemmini: $(FPGA_EXE_GEMMINI)
	$(UART_TSI) +tty=$(TTY) +baudrate=$(BAUDRATE) $(FPGA_EXE_GEMMINI)

# UART run (standard)
run-uart-std: $(FPGA_EXE_STD)
	$(UART_TSI) +tty=$(TTY) +baudrate=$(BAUDRATE) $(FPGA_EXE_STD)

# -----------------------------------------------------------------------------
# Cleanup
# -----------------------------------------------------------------------------
clean:
	rm -f $(TORCH_MLIR_FILE) \
	      step1_bufferized.mlir step2_memref.mlir step3_gemmini.mlir \
	      step4_metadata.mlir step5_loops.mlir step6_lowered.mlir \
	      step7_llvm.ll step3_loops.mlir step4_llvm_dialect.mlir \
	      step5_strided.mlir step6_llvm.ll \
	      $(OBJ_GEMMINI) $(OBJ_STD) \
	      $(EXE_GEMMINI) $(EXE_STD) \
	      $(FPGA_EXE_GEMMINI) $(FPGA_EXE_STD)
