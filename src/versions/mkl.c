#include "matrix_utils.h"
#include <mkl.h>

// Import the parsed argument from main.c
extern int n_threads;

void multiply_matrices(int n, m_type (*restrict A)[n], m_type (*restrict B)[n], m_type (*restrict C)[n]) {
    // Tell oneMKL exactly how many threads to use
    #ifdef SEQUENTIAL
    // Sequential mode: force single-threaded MKL regardless of n_threads
    mkl_set_num_threads(1);
    #elif PARALLEL
    // Parallel mode: let MKL use the requested number of threads
    mkl_set_num_threads(n_threads);
    #else
        #error "Must define either SEQUENTIAL or PARALLEL"
    #endif


    double alpha = 1.0;
    double beta = 0.0;

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n, n, n, alpha, 
                (const double *)A, n, 
                (const double *)B, n, 
                beta, (double *)C, n);
}