## Overview of Directories

1. **`local_openblas/`**: Contains the local installation of the OpenBLAS library. Includes headers in `include/` and compiled `.so` libraries and CMake configurations in `lib/` used for OpenBLAS baseline benchmarks.

2. **`MPI/`**: Contains the distributed memory parallelization via Message Passing Interface (MPI).
   - `matmul_mpi_exec`: The executable for the MPI matrix multiplication.
   - `matmul_mpi.c`: Source code for the MPI implementation.
   - `mpi_benchmark_results.csv`: Exported result timings from the MPI execution tests.
   - `plots.ipynb`: Jupyter notebook for plotting MPI metrics (e.g., speedup, efficiency, throughput) and scalability analysis.
   - `run_tests.sh`: Bash script to submit/execute the MPI jobs on the HPC discrete multi-node clusters.

3. **`OPENMP/`**: Contains the shared memory parallelization files managed by OpenMP.
   - `openmp_plots/`: Python plotting scripts (`plot_openmp_binding.py`, `plot_openmp_scheduling.py`, etc.) used to plot OpenMP experimental metrics like chunk scheduling and thread binding.

4. **`SEQUENTIAL/`**: Contains files governing the baseline single-threaded CPU version.
   - `sequential_plots/`: Python plotting scripts focused on the single-core cache optimizations and sequential ICC compiler performance.

5. **`src/`**: Contains the core C source codes that link together the matrix operations and optimizations.
   - `arg_parser.c` & `arg_parser.h`: Standardized command line argument parsing utilities used universally.
   - `io_utils.c` & `io_utils.h`: Output and file generation helper functions.
   - `matrix_utils.c` & `matrix_utils.h`: General matrix memory allocators, initializers, and block managers.
   - `main.c`: General execution driver for matrix multiplications.
   - `original_implementation.c`: The baseline, non-optimized sequential generic matrix multiplication kernel.
   - `cuda/`: Scripts containing the device kernels for CUDA GPU execution.
   - `versions/`: Different matrix multiplication version branches implementing tiling, vectorization looping, or library wrappers (OpenBLAS/MKL).

6. **`TOOLS/`**: Contains the python orchestrator suite used to script experiments, orchestrate compiler builds, and run profiling.
   - For full tool documentation detailing parameters and architecture, **please see [TOOLS/docs](TOOLS/docs/)**.
   - `build.py`: Automates compiling different project implementations using different compilers (GCC/ICC).
   - `test.py`: The global test runner used for sweeping over iterations and collecting benchmarking metrics to `csv`.
   - `test_run.py`: A wrapper used to run the compiled executables via external diagnostic/profiling tools like Valgrind or Intel Advisor.
   - `plot.py`: General plotting script for generating visualizations from `.csv` baseline outputs.
   - `config.py`: Centralized configuration script for controlling the experimental matrix sizes and environments.

## Root Level Files

- **`README.md`**: Offers the overarching project goal, compilation recaps, and examples exhibiting how to reproduce outputs and build configurations.
- **`docs.md`**: This exact file - providing a comprehensive outline of the repository structure.