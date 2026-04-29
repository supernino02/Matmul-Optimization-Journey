#include "matrix_utils.h"

//return a matrix aligned to 64 bytes if ALIGN is defined
//otherwise a standard contiguous block
void *create_matrix(int n) {
    if  (n <= 0) {
        fprintf(stderr, "Matrix dimensions must be positive.\n");
        exit(EXIT_FAILURE);
    }

    size_t size = (size_t)n * n * sizeof(m_type);
    void *matrix;

#ifdef ALIGN
    if (posix_memalign(&matrix, LINE_SIZE, size) != 0) {
        exit(EXIT_FAILURE);
    }
#else
    matrix = malloc(size);
    if (!matrix) exit(EXIT_FAILURE);
#endif

    return matrix;

}

void free_matrix(void *matrix)
{
    free(matrix);
}


#ifdef ALIGN
void init_matrices(int n, int padded_n, m_type (*A)[padded_n], m_type (*B)[padded_n], m_type (*C)[padded_n])
{
    for (int i = 0; i < padded_n; i++) {
        for (int j = 0; j < padded_n; j++) {
            if (i < n && j < n) {
                A[i][j] = (m_type)rand() / RAND_MAX; //random value in [0,1]
                B[i][j] = (m_type)rand() / RAND_MAX; //random value in [0,1]
            } else {
                A[i][j] = 0.0;
                B[i][j] = 0.0;
            }
            C[i][j] = 0.0; //initialize C to zero
        }
    }
}
#else
void init_matrices(int n, m_type (*A)[n], m_type (*B)[n], m_type (*C)[n])
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i][j] = (m_type)rand() / RAND_MAX; //random value in [0,1]
            B[i][j] = (m_type)rand() / RAND_MAX; //random value in [0,1]
            C[i][j] = 0.0; //initialize C to zero
        }
    }
}
#endif