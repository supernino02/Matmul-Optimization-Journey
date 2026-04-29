#ifndef DEBUG_UTILS_H
#define DEBUG_UTILS_H

#include <stdio.h>
#include <math.h>
#include "matrix_utils.h"

typedef struct errorEntry {
    int row;
    int col;
    double expected;
    double actual;
} ErrorEntry;

typedef struct ResultEntry {
    double real_time;
    double cpu_time;
} ResultEntry;

// Print results in raw format to stdout
void printResult(ResultEntry results);

//find errors in the result matrix C -> naive cpu code

int get_errors(int n, m_type (*A)[n], m_type (*B)[n], m_type (*C)[n], double epsilon, int max_errors, ErrorEntry errors[]);
// Print errors to stderr and exit
void printErrors(ErrorEntry errors[], int num_errors);


#endif // DEBUG_UTILS_H
