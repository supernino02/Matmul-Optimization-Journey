#include "matrix_utils_cuda.h"

extern "C"
void multiply_matrices(int n, m_type * __restrict__ A,
                              m_type * __restrict__ B,
                              m_type * __restrict__ C)
{
    size_t size = (size_t)n * n * sizeof(m_type);
    m_type *dA, *dB, *dC;

    /* allocate device memory */
    CUDA_CHECK(cudaMalloc(&dA, size));
    CUDA_CHECK(cudaMalloc(&dB, size));
    CUDA_CHECK(cudaMalloc(&dC, size));

    /* copy host → device */
    CUDA_CHECK(cudaMemcpy(dA, A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B, size, cudaMemcpyHostToDevice));

    /* launch the kernel on the GPU */
    launch_kernel(n, dA, dB, dC);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    /* copy device → host */
    CUDA_CHECK(cudaMemcpy(C, dC, size, cudaMemcpyDeviceToHost));

    /* free device memory */
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
}
