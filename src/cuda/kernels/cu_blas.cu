#include "../matrix_utils_cuda.h"
#include <cublas_v2.h>

/* Global cuBLAS handle */
static cublasHandle_t handle = NULL;

/* ── Cleanup cuBLAS handle (optional, call at program end) ───────────── */
extern "C"
void cleanup_cublas() {
    if (handle != NULL) {
        cublasDestroy(handle);
        handle = NULL;
    }
}

/* ── Launch kernel using cuBLAS dgemm ──────────────────────────────── */
extern "C"
void launch_kernel(int n, m_type *dA, m_type *dB, m_type *dC) {
    
    // Create cuBLAS handle
    cublasHandle_t handle;
    CUDA_CHECK((cudaError_t)cublasCreate(&handle));
    
    // Setup parameters for C = alpha * A * B + beta * C
    const double alpha = 1.0;
    const double beta = 0.0;
    
    // Perform the matrix multiplication: C = alpha * A * B + beta * C
    CUDA_CHECK((cudaError_t)cublasDgemm(
        handle,
        CUBLAS_OP_N,  // Don't transpose A
        CUBLAS_OP_N,  // Don't transpose B
        n,            // Number of rows of matrix A and C
        n,            // Number of columns of matrix B and C
        n,            // Number of columns of A and rows of B
        &alpha,       // Scalar alpha
        dA,           // Matrix A
        n,            // Leading dimension of A
        dB,           // Matrix B
        n,            // Leading dimension of B
        &beta,        // Scalar beta
        dC,           // Matrix C
        n             // Leading dimension of C
    ));
    
    // Cleanup
    CUDA_CHECK((cudaError_t)cublasDestroy(handle));
}
