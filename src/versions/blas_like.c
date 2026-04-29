/* ================================================================
 * COMPILE-TIME GUARDS & INCLUDES
 * ================================================================ */
#include <immintrin.h> 
#include <string.h>    
#include "matrix_utils.h"

#ifdef OPENMP
#include <omp.h>
extern int n_threads;
extern omp_sched_t runtime_schedule_type;
extern int block_size;
#endif

#ifndef ALIGN
#error "ALIGN must be defined"
#endif
_Static_assert(sizeof(m_type) == sizeof(double), "M_TYPE must be double.");

#define MIN(x, y) ((x) < (y) ? (x) : (y))

#define MR 4
#define NR 8

#ifndef MC
#define MC 256 
#endif

#ifndef NC
#define NC 512
#endif

#ifndef KC
#define KC 256
#endif

// Define strictly typed 3D matrices for the packed buffers
// Dimensions: [Number of Panels][Depth of Panel][Width of Panel]
typedef double (*PackedA_t)[KC][MR];
typedef double (*PackedB_t)[KC][NR];

/* ================================================================
 * 1. PANEL DATA PACKING (Matrix-Mapped)
 * ================================================================ */
static inline void pack_A(int n, double (*A)[n], PackedA_t packedA, int i0, int k0, int mc, int kc) {
    for (int i = 0; i < mc; i += MR) {
        int panel_idx = i / MR;
        for (int p = 0; p < kc; p++) {
            #pragma unroll(4)
            for (int ir = 0; ir < MR; ir++) {
                packedA[panel_idx][p][ir] = A[i0 + i + ir][k0 + p];
            }
        }
    }
}

static inline void pack_B(int n, double (*B)[n], PackedB_t packedB, int k0, int j0, int kc, int nc) {
    for (int j = 0; j < nc; j += NR) {
        int panel_idx = j / NR;
        for (int p = 0; p < kc; p++) {
            #pragma unroll(8)
            for (int jr = 0; jr < NR; jr++) {
                packedB[panel_idx][p][jr] = B[k0 + p][j0 + j + jr];
            }
        }
    }
}

/* ================================================================
 * 2. THE 4x8 AVX2 MICRO-KERNEL
 * ================================================================ */
static inline void kernel_4x8(int n, double (*C)[n], 
                              double packedA_panel[KC][MR], 
                              double packedB_panel[KC][NR], 
                              int kc, int i0, int j0) {
    __m256d c[MR][2]; 

    #pragma unroll(MR)
    for (int i = 0; i < MR; i++) {
        c[i][0] = _mm256_setzero_pd();
        c[i][1] = _mm256_setzero_pd();
    }

    for (int p = 0; p < kc; p++) {
        // Aligned loads directly from the 2D panel representation
        __m256d b0 = _mm256_load_pd(&packedB_panel[p][0]);
        __m256d b1 = _mm256_load_pd(&packedB_panel[p][4]);

        #pragma unroll(MR)
        for (int i = 0; i < MR; i++) {
            __m256d a = _mm256_broadcast_sd(&packedA_panel[p][i]);
            c[i][0] = _mm256_fmadd_pd(a, b0, c[i][0]);
            c[i][1] = _mm256_fmadd_pd(a, b1, c[i][1]);
        }
    }

    #pragma unroll(MR)
    for (int i = 0; i < MR; i++) {
        __m256d orig0 = _mm256_loadu_pd(&C[i0 + i][j0 + 0]);
        __m256d orig1 = _mm256_loadu_pd(&C[i0 + i][j0 + 4]);
        
        _mm256_storeu_pd(&C[i0 + i][j0 + 0], _mm256_add_pd(orig0, c[i][0]));
        _mm256_storeu_pd(&C[i0 + i][j0 + 4], _mm256_add_pd(orig1, c[i][1]));
    }
}

/* ================================================================
 * 3. MAIN MATMUL LOOP
 * ================================================================ */
void multiply_matrices(int n, double (*restrict A)[n], double (*restrict B)[n], double (*restrict C)[n]) {
    //hint for optimizations
    A = __builtin_assume_aligned(A, LINE_SIZE);
    B = __builtin_assume_aligned(B, LINE_SIZE);
    C = __builtin_assume_aligned(C, LINE_SIZE);

    #ifdef OPENMP
    omp_set_schedule(runtime_schedule_type, block_size);
    omp_set_num_threads(n_threads);
    #pragma omp parallel shared(A, B, C, n) 
    #endif
    {
        const PackedA_t packedA = (PackedA_t)_mm_malloc(MC * KC * sizeof(double), LINE_SIZE); //64
        const PackedB_t packedB = (PackedB_t)_mm_malloc(NC * KC * sizeof(double), LINE_SIZE); //64

        #ifdef OPENMP
        //the scheduling is choosen at runtime
        #pragma omp for schedule(runtime)
        #endif
        for (int j = 0; j < n; j += NC) {
            int nc = MIN(NC, n - j);
            
            for (int k = 0; k < n; k += KC) {
                int kc = MIN(KC, n - k);
                
                pack_B(n, B, packedB, k, j, kc, nc);
                
                for (int i = 0; i < n; i += MC) {
                    int mc = MIN(MC, n - i);
                    
                    pack_A(n, A, packedA, i, k, mc, kc);

                    for (int ir = 0; ir < mc; ir += MR) {
                        for (int jr = 0; jr < nc; jr += NR) {
                            // Pass the specific 2D micro-panels directly
                            kernel_4x8(n, C,
                                    packedA[ir / MR],
                                    packedB[jr / NR],
                                    kc, i + ir, j + jr);
                        }
                    }
                }
            }
        }

        _mm_free(packedA);
        _mm_free(packedB);
    }
}