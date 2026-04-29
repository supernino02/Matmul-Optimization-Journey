#include "../matrix_utils_cuda.h"

//allow to set cu_block_x, cu_block_y at runtime (instead of fixed at compile time)
extern int cu_block_x, cu_block_y;

/* ── Kernel ─────────────────────────────────────────────────────────── */

__global__ void kernel(int n, const m_type *A, const m_type *B, m_type *C) {
    // compute position in C that this thread is responsible for
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    // `if` condition is necessary for when n isn't a multiple of blockDim.
    if (x < n && y < n) {
        m_type tmp = 0.0;
        for (int i = 0; i < n; ++i) {
            tmp += A[x * n + i] * B[i * n + y];
        }
        // C = C + A@B
        C[x * n + y] = tmp;
    }
}

/* ── Launch (called by cuda_wrapper.cu) ─────────────────────────────── */

extern "C"
void launch_kernel(int n, m_type *dA, m_type *dB, m_type *dC) {
    dim3 block(cu_block_x, cu_block_y);
    dim3 grid(CEIL_DIV(n, cu_block_x), CEIL_DIV(n, cu_block_y));
    kernel<<<grid, block>>>(n, dA, dB, dC);
}
