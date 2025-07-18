#include <cuda_runtime.h>



#ifdef __cplusplus
#define __convention__ extern "C"
#endif

// single use matmul
__convention__ void matmul_cuda(const float* A, size_t sizeA_bytes, int Avs, int Ahs, int Avm, int Ahm,
                 const float* B, size_t sizeB_bytes, int Bvs, int Bhs, int Bvm, int Bhm,
                 float* C, int V, int N, int H);

// must allocate d_A first and free return value after
__convention__ float* matmul_cuda_flow(const float* d_A, size_t sizeA_bytes, int Avs, int Ahs, int Avm, int Ahm,
                 const float* B, size_t sizeB_bytes, int Bvs, int Bhs, int Bvm, int Bhm,
                 float* C, int V, int N, int H);

__convention__ float* alloc_cuda(const float* A, size_t sizeA_bytes, int Avs, int Ahs, int Avm, int Ahm);

__convention__ void free_cuda(float* d_C);