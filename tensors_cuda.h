#include <cuda_runtime.h>

__global__ void matmul_kernel(
    float* A, float* B, float* C,
    int V, int H, int N,
    int A_v_mult, int A_h_mult,
    int B_v_mult, int B_h_mult);