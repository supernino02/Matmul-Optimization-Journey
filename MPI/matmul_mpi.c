#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef N
#define N 5000
#endif

#ifndef ITER
#define ITER 3
#endif

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Allocazione esatta come nel report (Puntatori ad array 2D)
    double (*A)[N] = NULL;
    double (*C)[N] = NULL;
    double (*B)[N] = malloc(sizeof(double[N][N]));

    int base_rows = N / size;
    int remainder = N % size;
    int my_rows = base_rows + (rank < remainder ? 1 : 0);

    double (*local_A)[N] = malloc(sizeof(double[my_rows][N]));
    double (*local_C)[N] = malloc(sizeof(double[my_rows][N]));

    int *sendcounts = malloc(size * sizeof(int));
    int *displs = malloc(size * sizeof(int));

    int current_displ = 0;
    for (int i = 0; i < size; i++) {
        int r = base_rows + (i < remainder ? 1 : 0);
        sendcounts[i] = r * N;
        displs[i] = current_displ;
        current_displ += sendcounts[i];
    }

    if (rank == 0) {
        A = malloc(sizeof(double[N][N]));
        C = malloc(sizeof(double[N][N]));
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A[i][j] = 2.0;
                B[i][j] = 3.0;
                C[i][j] = 0.0;
            }
        }
    }

    double total_time = 0.0;

    for (int iter = 0; iter < ITER; iter++) {
        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();

        MPI_Scatterv(A, sendcounts, displs, MPI_DOUBLE, local_A, my_rows * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(B, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Ciclo di calcolo esattamente fedele all'originale
        for (int i = 0; i < my_rows; i++) {
            for (int j = 0; j < N; j++) {
                local_C[i][j] = 0.0;
            }
        }

        for (int i = 0; i < my_rows; i++) {
            for (int k = 0; k < N; k++) {
                for (int j = 0; j < N; j++) {
                    local_C[i][j] += local_A[i][k] * B[k][j];
                }
            }
        }

        MPI_Gatherv(local_C, my_rows * N, MPI_DOUBLE, C, sendcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
        double end = MPI_Wtime();
        
        if (rank == 0) {
            total_time += (end - start);
        }
    }

    if (rank == 0) {
        printf("RESULT,%d,%d,%0.6f\n", N, size, total_time / ITER);
    }

    free(sendcounts); free(displs); free(B); free(local_A); free(local_C);
    if (rank == 0) { free(A); free(C); }

    MPI_Finalize();
    return 0;
}