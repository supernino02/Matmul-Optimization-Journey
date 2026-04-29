#include "../matrix_utils.h"

#ifdef OPENMP
#include <omp.h>
extern int n_threads;
extern omp_sched_t runtime_schedule_type;
extern int block_size;
#endif

void multiply_matrices(int n, m_type (*restrict A)[n], m_type (*restrict B)[n], m_type (*restrict C)[n]) {
    
    #ifdef ALIGN
        A = __builtin_assume_aligned(A, LINE_SIZE);
        B = __builtin_assume_aligned(B, LINE_SIZE);
        C = __builtin_assume_aligned(C, LINE_SIZE);
    #endif

    #ifdef OPENMP
    //schedule and number of threads for parallel region
    //choosen at runtime
    omp_set_schedule(runtime_schedule_type, block_size);
    omp_set_num_threads(n_threads);
    #pragma omp parallel for   \
        shared(A, B, C, n)     \
        schedule(runtime)      
    //binding choosen at compile time
    #endif
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; k++) {
            for (int j = 0; j < n; ++j) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}