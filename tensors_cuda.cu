#include "tensors_cuda.h"

__global__ void matmul_kernel(
    float* A, float* B, float* C,
    int V, int H, int N,
    int A_v_mult, int A_h_mult,
    int B_v_mult, int B_h_mult) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < V && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < H; ++k) {
            float a = A[(row + A_v_mult) * H + (k + A_h_mult)];
            float b = B[(k + B_v_mult) * N + (col + B_h_mult)];
            acc += a * b;
        }
        C[row * N + col] = acc;
    }
}
