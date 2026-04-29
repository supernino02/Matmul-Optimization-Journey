#ifndef MATRIX_UTILS_CUDA_H
#define MATRIX_UTILS_CUDA_H

#include <stdio.h>
#include <stdlib.h>

/* ── CUDA error-checking macro ──────────────────────────────────────── */
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

//helper macro for ceiling division to compute grid dimensions
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))


typedef double m_type;

/* ── C-linkage prototypes ───────────────────────────────────────────── */
extern "C" {
    /* from the C sources (matrix_utils.c, main.c, …) */
    void *create_matrix(int n);
    void  free_matrix(void *matrix);
    void  init_matrices(int n, m_type *A, m_type *B, m_type *C);

    /* implemented in cuda_wrapper.cu */
    void multiply_matrices(int n, m_type * __restrict__ A,
                                m_type * __restrict__ B,
                                m_type * __restrict__ C);

    /*
    * Implemented in each cuda_N.cu kernel file.
    * cuda_wrapper.cu calls this — swap the .cu file to change the kernel.
    */
    void launch_kernel(int n, m_type *dA, m_type *dB, m_type *dC);
}

#endif /* MATRIX_UTILS_CUDA_H */
