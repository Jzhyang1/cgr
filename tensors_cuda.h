#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C"
#endif

void matmul_cuda(const float* A, size_t sizeA_bytes, int Avs, int Ahs, int Avm, int Ahm,
                 const float* B, size_t sizeB_bytes, int Bvs, int Bhs, int Bvm, int Bhm,
                 float* C, int V, int N, int H);