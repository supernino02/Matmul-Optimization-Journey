# Matrix Multiplication Analysis & Parallelization

## Project Goal
The goal of this project is to analyze and optimize Matrix Multiplication performance through iterative parallelization. Matrix Multiplication is a fundamental high-performance computing (HPC) kernel characterized by high arithmetic intensity. Because of its complexity of $O(n^3)$, computing the matrix product serves as an excellent benchmark for understanding CPU cache mechanics, compiler optimizations, and parallel architecture paradigms.

## Recap
Starting from a reference sequential implementation (a naive nested loop approach), the project progressively introduces hardware-aware optimizations by exploiting the hierarchy of parallelism in modern HPC systems:
1. **Instruction and Vector Level (SIMD)**: At the core level via aggressive sequential compilation optimizations (e.g. GCC, ICC/ICX compiler flags) and cache-aware strategies like memory block tiling.
2. **Hyperthreading and Core Level (Shared Memory)**: Distributing the workload block across available threads and cores on a single CPU node using OpenMP.
3. **Accelerator Level (GPU)**: Offloading heavily data-parallel tasks to specialized hardware architectures using CUDA (e.g., with specific naive and tiling kernels on Nvidia GPUs).
4. **Node Level (Distributed Memory)**: Partitioning and distributing the matrix computation across discrete interconnected nodes using MPI.

The implementations are subsequently profiled, tuned, and compared against highly optimized standard libraries (MKL and OpenBLAS).

## Replicating the Results
You can use the provided Python tools to rebuild and run the experiments described in the final report. Here are some command line examples to replicate the workflows:

1. **Building Executables:**
   Use `TOOLS/build.py` to compile the different implementations.
   ```bash
   # Build the sequential baseline using GCC with O1 optimization
   python TOOLS/build.py --compiler gcc --version original --name gcc_1 -p "-O1"

   # Build the BLAS-like OpenMP version using max optimizations
   python TOOLS/build.py --compiler icx --version blas_like --name icx_blas_like_omp --best-flags
   ```

2. **Running Tests:**
   Use `TOOLS/test.py` to run parameter sweeps and performance benchmark loops on your built executables.
   ```bash
   # Run the gcc_1 build 3 times, writing the results to sequential_experiment
   python TOOLS/test.py --builds gcc_1 --iterations 3 --experiment sequential_experiment

   # Run multiple builds sequentially for a full performance comparison
   python TOOLS/test.py --builds gcc_1 icx_blas_like_omp --iterations 5 --experiment full_comparison
   ```

3. **Running Profilers:**
   Use `TOOLS/test_run.py` to profile your executable's cache behavior or performance.
   ```bash
   # Profile cache performance / collisions using Valgrind
   python TOOLS/test_run.py --valgrind --build gcc_1 --experiment cache_profiling

   # Profile execution with Intel Advisor for Roofline analysis
   python TOOLS/test_run.py --intel-advisor --build icx_blas_like_omp --experiment roofline_analysis
   ```