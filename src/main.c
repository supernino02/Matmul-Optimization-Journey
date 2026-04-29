#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include <omp.h>

#include "arg_parser.h"
#include "io_utils.h"
#include "matrix_utils.h"

//if N is given at compile time, ignore command line argument and use the compile time value
#ifdef N_CONST
    #define N_orig ((int)N_CONST)
    #ifdef ALIGN
        #define N PAD_N((int)N_CONST)
    #else
        #define N ((int)N_CONST)
    #endif
#endif

//global variables for parallelization parameters (if needed)
#ifdef OPENMP
    //activate the parallelization flag
    #define PARALLEL
    omp_sched_t runtime_schedule_type;
    int block_size;
#endif

//for openmp, mkl and openblas
#ifdef PARALLEL
    int n_threads;  
#endif

//global variables for cuda parameters (if needed)
#ifdef CUDA
    int cu_block_x, cu_block_y;
#endif

//errors default values
#define EPSILON 0.000001
#define MAX_ERRORS 32

int main(int argc, char *argv[]) {
    //--------------------//
    //  PARSE ARGUMENTS   //
    //--------------------//
    ArgParser *parser = ArgParser_create(argc, argv);

    //verification parameters always possible to set
    const bool verify_flag = ArgParser_getFlag(parser, "-verify");
    const double epsilon = ArgParser_getDouble(parser, "--epsilon", EPSILON);
    const int max_errors = ArgParser_getInt(parser, "--max-errors", MAX_ERRORS);

    //if -verify is not set, ignore epsilon and max-errors and warn user if they are set
    if (!verify_flag && (epsilon != EPSILON || max_errors != MAX_ERRORS)) {
        fprintf(stderr, "Warning: --epsilon and --max-errors are ignored if -verify flag is not set.\n");
        exit(EXIT_FAILURE);
    }

    //-----MATRIX SIZE-----//
    #ifndef N_CONST
    int N_orig = ArgParser_getInt(parser, "--n", -1);
        #ifdef ALIGN
    //round up to nearest multiple of cache line elements if ALIGN is defined
    int N = PAD_N(N_orig);
        #else
    int N = N_orig;
        #endif
    #endif

    //-----HOW MANY THREADS-----//
    #ifdef PARALLEL
        n_threads = ArgParser_getInt(parser, "--threads", omp_get_max_threads()); //default max available
    #endif

    //-----OPENMP PARAMETERS-----//
    #ifdef OPENMP
    block_size = ArgParser_getInt(parser, "--block-size", 0);//by default, let openmp decide
    runtime_schedule_type = (omp_sched_t)ArgParser_getInt(parser, "--schedule", 1);
    //1 = omp_sched_static,2 = omp_sched_dynamic,3 = omp_sched_guided,4 = omp_sched_auto
    #endif

    //-----CUDA PARAMETERS-----//
    #ifdef CUDA
    cu_block_x = ArgParser_getInt(parser, "--cu-block-x", 16);
    cu_block_y = ArgParser_getInt(parser, "--cu-block-y", 16);
    #endif

    //destroy parser, notify unused arguments
    ArgParser_destroy(parser);

    //--------------------//
    //  MALLOC STRUCTURES //
    //--------------------//
    m_type (*A)[N] = create_matrix(N);
    m_type (*B)[N] = create_matrix(N);
    m_type (*C)[N] = create_matrix(N);

    // fill matrices (RAND,RAND,0)
    #ifdef ALIGN
    init_matrices(N_orig, N, A, B, C);
    #else
    init_matrices(N, A, B, C);
    #endif

    //storage for single result
    ResultEntry results;

    // start time
    double real_t1 = omp_get_wtime();
    clock_t cpu_t1 = clock();

    //----BENCHMARK ONLY THIS----//
    multiply_matrices(N, A, B, C);

    // end time
    double real_t2 = omp_get_wtime();
    clock_t cpu_t2 = clock();

    // store results
    results.real_time = real_t2 - real_t1;
    results.cpu_time = (double)(cpu_t2 - cpu_t1) / CLOCKS_PER_SEC;

    // correctness check if requested
    if (verify_flag) {

        //print verification info
        #ifdef ALIGN
        fprintf(stderr,"Matrix size: %d x %d (padded to %d x %d)\n", N_orig, N_orig, N, N);
        #else
        fprintf(stderr,"Matrix size: %d x %d\n", N, N);
        #endif
        fprintf(stderr,"Verifying results with epsilon = %e and max errors to print = %d\n", epsilon, max_errors);

        //allocate error entries array
        ErrorEntry *error_entries = malloc(sizeof(ErrorEntry) * max_errors);

        //get errors
        int num_errors = get_errors(N, A, B, C, epsilon, max_errors, error_entries);
        if (num_errors) {
            int tot_errors = num_errors < max_errors ? num_errors : max_errors; //print only max_errors if there are more
            printErrors(error_entries, tot_errors);
            exit(EXIT_FAILURE);
        }
        else {
            fprintf(stderr, "No errors found.\n");
        }

        free(error_entries);
    }

    // print raw result to stdout
    printResult(results);
    
    //--------------------//
    // GARBAGE COLLECTING //
    //--------------------//
    free_matrix(A);
    free_matrix(B);
    free_matrix(C);

    exit(EXIT_SUCCESS);
}
