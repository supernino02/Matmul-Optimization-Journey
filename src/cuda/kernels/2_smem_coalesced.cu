#include "../matrix_utils_cuda.h"

#define BLOCKSIZE_COAL 16

/* ── Kernel ─────────────────────────────────────────────────────────── */

__global__ void kernel(int n, const m_type *A, const m_type *B, m_type *C) {
    const int x = blockIdx.x * BLOCKSIZE_COAL + (threadIdx.x / BLOCKSIZE_COAL);
    const int y = blockIdx.y * BLOCKSIZE_COAL + (threadIdx.x % BLOCKSIZE_COAL);
    if (x < n && y < n) {
        m_type tmp = 0.0;
        for (int i = 0; i < n; ++i) tmp += A[x * n + i] * B[i * n + y];
        C[x * n + y] = tmp;
    }
}
/* ── Launch (called by cuda_wrapper.cu) ─────────────────────────────── */

extern "C"
void launch_kernel(int n, m_type *dA, m_type *dB, m_type *dC) {
    dim3 block(BLOCKSIZE_COAL *  BLOCKSIZE_COAL);
    dim3 grid(CEIL_DIV(n, BLOCKSIZE_COAL), CEIL_DIV(n, BLOCKSIZE_COAL));
    kernel<<<grid, block>>>(n, dA, dB, dC);
}
