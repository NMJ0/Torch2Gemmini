SHELL := /bin/bash
ROOT_DIR ?= $(abspath ..)
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

PY_SCRIPT := generate_mlir.py
TORCH_MLIR_FILE := matmul_from_torch.mlir

OBJ_GEMMINI := log_gemmini.o
OBJ_STD     := log_std.o

WRAPPER := wrapper.c
EXE_GEMMINI := matmul_gemmini.elf
EXE_STD     := matmul_std.elf
FPGA_EXE_GEMMINI := matmul-fpga_gemmini.elf
FPGA_EXE_STD     := matmul-fpga_std.elf

TORCH_MLIR_SRC ?= $(ROOT_DIR)/torch-mlir
TORCH_MLIR_BUILD ?= $(TORCH_MLIR_SRC)/build
PYTHONPATH_TORCH_MLIR := $(TORCH_MLIR_SRC)/projects/pt1/python:$(TORCH_MLIR_SRC)/python:$(TORCH_MLIR_BUILD)/tools/torch-mlir/python_packages/torch_mlir

.PHONY: all all-std gen_mlir lower-gemmini lower-std link-gemmini link-std \
	run-gemmini run-std run-uart-gemmini run-uart-std clean

all: $(EXE_GEMMINI)

all-std: $(EXE_STD)

gen_mlir: $(TORCH_MLIR_FILE)

$(TORCH_MLIR_FILE): $(PY_SCRIPT)
	conda run -n nmj0 env PYTHONPATH=$(PYTHONPATH_TORCH_MLIR) \
	$(TORCH_MLIR_PYTHON) $(PY_SCRIPT)

lower-gemmini: $(OBJ_GEMMINI)

$(OBJ_GEMMINI): $(TORCH_MLIR_FILE)
	@echo "=== Gemmini lowering ==="
	$(BUDDY_OPT) ./$(TORCH_MLIR_FILE) \
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

lower-std: $(OBJ_STD)

$(OBJ_STD): $(TORCH_MLIR_FILE)
	@echo "=== Gemmini lowering ==="
	$(BUDDY_OPT) ./$(TORCH_MLIR_FILE) \
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

link-gemmini: $(EXE_GEMMINI)

$(EXE_GEMMINI): $(WRAPPER) $(OBJ_GEMMINI)
	$(RISCV_GCC) $(WRAPPER) $(OBJ_GEMMINI) -O2 -static -o $(EXE_GEMMINI)

link-std: $(EXE_STD)

$(EXE_STD): $(WRAPPER) $(OBJ_STD)
	$(RISCV_GCC) $(WRAPPER) $(OBJ_STD) -O2 -static -o $(EXE_STD)

run-gemmini: $(EXE_GEMMINI)
	$(SPIKE) --extension=gemmini pk $(EXE_GEMMINI)

run-std: $(EXE_STD)
	$(SPIKE)  pk $(EXE_STD)

link-fpga-gemmini: $(FPGA_EXE_GEMMINI)

$(FPGA_EXE_GEMMINI): $(WRAPPER) $(OBJ_GEMMINI)
	$(CHIPYARD_GCC) -g -static -nostartfiles -march=rv64imafd -mabi=lp64d -mcmodel=medany -specs=nosys.specs \
	  -fPIC \
	  -T $(CHIPYARD_BAREMETAL)/glossy/glossy.ld \
	  $(WRAPPER) $(OBJ_GEMMINI) -o $(FPGA_EXE_GEMMINI) \
	  -L$(CHIPYARD_BUILD)/glossy \
	  -Wl,--whole-archive $(CHIPYARD_BUILD)/glossy/libglossy.a -Wl,--no-whole-archive \
	  $(CHIPYARD_BUILD)/platform/default/libchip-config.a \
	  $(CHIPYARD_BUILD)/driver/rocket-chip/rocketcore/librocketcore.a \
	  $(CHIPYARD_BUILD)/driver/rocket-chip/clint/libclint.a \
	  $(CHIPYARD_BUILD)/driver/rocket-chip/plic/libplic.a \
	  $(CHIPYARD_BUILD)/driver/htif/libhtif.a

link-fpga-std: $(FPGA_EXE_STD)

$(FPGA_EXE_STD): $(WRAPPER) $(OBJ_STD)
	$(CHIPYARD_GCC) -g -static -nostartfiles -march=rv64imafd -mabi=lp64d -mcmodel=medany -specs=nosys.specs \
	  -fPIC \
	  -T $(CHIPYARD_BAREMETAL)/glossy/glossy.ld \
	  $(WRAPPER) $(OBJ_STD) -o $(FPGA_EXE_STD) \
	  -L$(CHIPYARD_BUILD)/glossy \
	  -Wl,--whole-archive $(CHIPYARD_BUILD)/glossy/libglossy.a -Wl,--no-whole-archive \
	  $(CHIPYARD_BUILD)/platform/default/libchip-config.a \
	  $(CHIPYARD_BUILD)/driver/rocket-chip/rocketcore/librocketcore.a \
	  $(CHIPYARD_BUILD)/driver/rocket-chip/clint/libclint.a \
	  $(CHIPYARD_BUILD)/driver/rocket-chip/plic/libplic.a \
	  $(CHIPYARD_BUILD)/driver/htif/libhtif.a

run-uart-gemmini: $(FPGA_EXE_GEMMINI)
	$(UART_TSI) +tty=$(TTY) +baudrate=$(BAUDRATE) $(FPGA_EXE_GEMMINI)

run-uart-std: $(FPGA_EXE_STD)
	$(UART_TSI) +tty=$(TTY) +baudrate=$(BAUDRATE) $(FPGA_EXE_STD)

clean:
	rm -f $(TORCH_MLIR_FILE) \
	      step1_bufferized.mlir step2_memref.mlir step3_gemmini.mlir \
	      step4_metadata.mlir step5_loops.mlir step6_lowered.mlir \
	      step7_llvm.ll step3_loops.mlir step4_llvm_dialect.mlir \
	      step5_strided.mlir step6_llvm.ll \
	      $(OBJ_GEMMINI) $(OBJ_STD) \
	      $(EXE_GEMMINI) $(EXE_STD) \
	      $(FPGA_EXE_GEMMINI) $(FPGA_EXE_STD)
