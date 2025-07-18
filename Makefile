# Variables
CXX := g++-14
NVCC := nvcc
CUDA_PATH := /usr/local/cuda-12.9
CXXFLAGS := -std=gnu++23 -fPIC -I$(CUDA_PATH)/include
NVCCFLAGS := -ccbin /usr/bin/gcc-14 -std=c++17 -Xcompiler -fPIC
LDFLAGS := -L$(CUDA_PATH)/lib64 -lcudart
BUILD_DIR := build
BIN_DIR := bin

# Source files
TRAIN_SRC := train.cxx
gpu_SRC := tensors_cuda.cu
EVAL_SRC := eval.cxx
TEST_SRC := test.cxx
EXAMPLE_SRC := example.cxx

# Output files
TRAIN_OBJ := $(BUILD_DIR)/train.o
GPU_OBJ := $(BUILD_DIR)/tensors_cuda.o
TRAIN_BIN := $(BIN_DIR)/train.a
EVAL_BIN := $(BIN_DIR)/eval.a
TEST_BIN := $(BIN_DIR)/test.a
EXAMPLE_BIN := $(BIN_DIR)/example.a

.PHONY: all main clean

all: main

main: train.a

# Pattern rule for building .a binaries from .cxx files, always depending on GPU_OBJ
%.a: %.cxx $(GPU_OBJ) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -g $< $(LDFLAGS) $(GPU_OBJ) -o $(BIN_DIR)/$@

# GPU object file
$(GPU_OBJ): $(gpu_SRC) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Train object file (not needed for pattern rule, but kept for main)
$(TRAIN_OBJ): $(TRAIN_SRC) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BIN_DIR) $(BUILD_DIR):
	mkdir -p $@

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)