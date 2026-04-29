#ifndef MATRIX_UTILS_H
#define MATRIX_UTILS_H

#include <stdio.h>
#include <stdlib.h>

#ifndef M_TYPE
#define M_TYPE double
#endif

typedef M_TYPE m_type;

// PAD_N rounds up to the nearest multiple of cache line elements
#ifdef ALIGN
#define LINE_SIZE 64 
#define PAD_N(x) ((int)(((x) + (LINE_SIZE / sizeof(m_type) - 1)) & ~(LINE_SIZE / sizeof(m_type) - 1)))
#else
#define PAD_N(x) (x)
#endif

/* Allocate an n x n matrix as a contiguous block.
   Returns a pointer suitable for use as matrix[i][j].
   If ALIGN is defined at compile time, uses aligned_alloc for cache-line alignment. */
void *create_matrix(int n);

/* Free a matrix previously returned by create_matrix */
void free_matrix(void *matrix);

/* Initialize matrices */
#ifdef ALIGN
void init_matrices(int n, int padded_n, m_type (*A)[padded_n], m_type (*B)[padded_n], m_type (*C)[padded_n]);
#else
void init_matrices(int n, m_type (*A)[n], m_type (*B)[n], m_type (*C)[n]);
#endif

void multiply_matrices(int n, m_type (*restrict A)[n], m_type (*restrict B)[n], m_type (*restrict C)[n]);

#endif // MATRIX_UTILS_H
