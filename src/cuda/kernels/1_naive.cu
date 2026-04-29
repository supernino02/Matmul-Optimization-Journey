#include "../matrix_utils_cuda.h"

//allow to set cu_block_x, cu_block_y at runtime (instead of fixed at compile time)
extern int cu_block_x, cu_block_y;

/* ── Kernel ─────────────────────────────────────────────────────────── */

__global__ void kernel(int n, const m_type *A, const m_type *B, m_type *C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        m_type sum = 0.0;
        for (int t = 0; t < n; ++t) sum += A[row * n + t] * B[t * n + col];
        C[row * n + col] = sum;
    }
}

/* ── Launch (called by cuda_wrapper.cu) ─────────────────────────────── */

extern "C"
void launch_kernel(int n, m_type *dA, m_type *dB, m_type *dC) {
    dim3 block(cu_block_x, cu_block_y);
    dim3 grid(CEIL_DIV(n, cu_block_x), CEIL_DIV(n, cu_block_y));
    kernel<<<grid, block>>>(n, dA, dB, dC);
}