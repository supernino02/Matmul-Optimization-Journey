#include <stdlib.h>
#include "io_utils.h"

// Print the results in a simple key=value format for easy parsing
void printResult(ResultEntry results) {
    printf("real_time_seconds=%f\n", results.real_time);
    printf("cpu_time_seconds=%f\n", results.cpu_time);
    fflush(stdout);
}

//check for errors in the result matrix C
int get_errors(int n, 
    m_type (*A)[n], m_type (*B)[n], m_type (*C)[n],
    double epsilon, int max_errors, ErrorEntry error_entries[])
{

    m_type (*Correct)[n] = create_matrix(n);

    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            Correct[i][k] = 0.0; //initialize Correct to zero
            for (int j = 0; j < n; j++) {
                Correct[i][k] += A[i][j] * B[j][k];
            } 
        }
    }

    int err_count = 0;
    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++)
            if (fabs(C[i][k] - Correct[i][k]) > epsilon) {
                if (err_count < max_errors) {
                    error_entries[err_count].row = i;
                    error_entries[err_count].col = k;
                    error_entries[err_count].expected = Correct[i][k];
                    error_entries[err_count].actual = C[i][k];
                }
                err_count++;
            }

    return err_count;
}

void printErrors(ErrorEntry errors[], int num_errors) {
    for (int i = 0; i < num_errors; ++i)
    {
        fprintf(stderr, "Error %d: C[%d,%d] = %.12f, expected %.12f\n",
                i + 1,
                errors[i].row,
                errors[i].col,
                errors[i].actual,
                errors[i].expected);
    }
    fprintf(stderr, "FAILED EXECUTION: %d errors\n", num_errors);
    exit(EXIT_FAILURE);
}