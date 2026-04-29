#include "../matrix_utils_cuda.h"

#define TILE_WIDTH 16

__global__ void kernel(int n, const m_type *A, const m_type *B, m_type *C) {
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    __shared__ m_type sA[TILE_WIDTH][TILE_WIDTH];
    __shared__ m_type sB[TILE_WIDTH][TILE_WIDTH];

    m_type sum = 0.0;
    int numTiles = (n + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int m = 0; m < numTiles; ++m) {
        if (row < n && m * TILE_WIDTH + tx < n)
            sA[ty][tx] = A[row * n + m * TILE_WIDTH + tx];
        else
            sA[ty][tx] = 0.0;

        if (m * TILE_WIDTH + ty < n && col < n)
            sB[ty][tx] = B[(m * TILE_WIDTH + ty) * n + col];
        else
            sB[ty][tx] = 0.0;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            sum += sA[ty][k] * sB[k][tx];
        }

        __syncthreads();
    }

    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}
/* ── Launch (called by cuda_wrapper.cu) ─────────────────────────────── */

extern "C"
void launch_kernel(int n, m_type *dA, m_type *dB, m_type *dC) {
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((n + TILE_WIDTH - 1) / TILE_WIDTH, (n + TILE_WIDTH - 1) / TILE_WIDTH);
    kernel<<<grid, block>>>(n, dA, dB, dC);
}
