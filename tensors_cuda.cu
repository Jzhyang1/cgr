#include "tensors_cuda.h"

__device__ inline int matrix_index(int v, int h, int v_start, int h_start, int v_mult, int h_mult) {
    return (v_start + v) * v_mult + (h_start + h) * h_mult;
}

// starts and ends passed in; C must be a full matrix
__global__ void matmul_kernel(const float* A, int Avs, int Ahs, int Avm, int Ahm,
                              const float* B, int Bvs, int Bhs, int Bvm, int Bhm,
                              float* C, int V, int N, int H) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < V && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < H; ++k) {
            int a_idx = matrix_index(row, k, Avs, Ahs, Avm, Ahm);
            int b_idx = matrix_index(k, col, Bvs, Bhs, Bvm, Bhm);
            sum += A[a_idx] * B[b_idx];
        }
        C[row * N + col] = sum;  // Assume C is contiguous in row-major layout
    }
}

void matmul_cuda(const float* A, size_t sizeA_bytes, int Avs, int Ahs, int Avm, int Ahm,
                 const float* B, size_t sizeB_bytes, int Bvs, int Bhs, int Bvm, int Bhm,
                 float* C, int V, int N, int H) {
    float *d_A, *d_B, *d_C;
    size_t sizeC = V * N * sizeof(float);

    cudaMalloc(&d_A, sizeA_bytes);
    cudaMalloc(&d_B, sizeB_bytes);
    cudaMalloc(&d_C, sizeC);

    cudaMemcpy(d_A, A, sizeA_bytes, cudaMemcpyKind::cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeB_bytes, cudaMemcpyKind::cudaMemcpyHostToDevice);

    dim3 threads(32, 32);
    dim3 blocks((N + 31) / 32, (V + 31) / 32);

    matmul_kernel<<<blocks, threads>>>(
        d_A, Avs, Ahs, Avm, Ahm,
        d_B, Bvs, Bhs, Bvm, Bhm,
        d_C, V, N, H
    );

    cudaMemcpy(C, d_C, sizeC, cudaMemcpyKind::cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}