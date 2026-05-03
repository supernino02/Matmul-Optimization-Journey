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

## Results

### Experimental Setup
All results were collected using **double-precision (FP64)** floating-point operations. To prevent the compiler from applying static, size-specific optimizations (such as loop unrolling or dead-code elimination), the matrix dimension $N$ was intentionally provided at runtime rather than as a compile-time constant.

- **CPU Experiments (Sequential, OpenMP, MPI):** Performed on workstations featuring a hybrid architecture with 8 Performance cores and 8 Efficient cores (24 logical threads total). The fastest OpenMP results were achieved using $T=16$ threads.
- **GPU Experiments (CUDA):** Conducted via Google Colab using an NVIDIA Tesla T4 GPU (Turing architecture, 2560 CUDA cores).
- **Metrics:** Computational throughput is measured in **GFLOPS**. The percentages ($\%$) reflect the throughput relative to the absolute best baseline measured in the study: Intel MKL running with OpenMP ($T=16$).

| Implementation | Time (s) <br> N=5000 | GFLOPS <br> N=5000 | % <br> N=5000 | &#124; | Time (s) <br> N=10000 | GFLOPS <br> N=10000 | % <br> N=10000 | &#124; | Time (s) <br> N=15000 | GFLOPS <br> N=15000 | % <br> N=15000 |
|---|---|---|---|:-:|---|---|---|:-:|---|---|---|
| **Sequential Implementations** | | | | &#124; | | | | &#124; | | | |
| ICC `-O0` | 388.781 | 0.64 | 0.14% | &#124; | 3028.552 | 0.66 | 0.13% | &#124; | - | - | - |
| ICC `-O1` | 69.403 | 3.60 | 0.77% | &#124; | 525.154 | 3.81 | 0.73% | &#124; | - | - | - |
| ICC `-O2` | 50.030 | 5.00 | 1.07% | &#124; | 401.861 | 4.98 | 0.96% | &#124; | - | - | - |
| ICC `-O3` | 18.258 | 13.69 | 2.94% | &#124; | 196.361 | 10.19 | 1.96% | &#124; | - | - | - |
| ICC `-O3 -xHost` | 9.901 | 25.25 | 5.42% | &#124; | 70.019 | 28.56 | 5.51% | &#124; | 275.428 | 24.51 | 4.69% |
| ICC `-O3 -xHost -DALIGN` | 6.511 | 38.40 | 8.24% | &#124; | 52.206 | 38.31 | 7.39% | &#124; | 175.374 | 38.49 | 7.37% |
| Custom OpenBLAS (Seq) | 4.556 | 54.87 | 11.78% | &#124; | 36.864 | 54.25 | 10.46% | &#124; | 123.927 | 54.47 | 10.43% |
| OpenBLAS (Seq) | 4.406 | 56.74 | 12.18% | &#124; | 32.458 | 61.62 | 11.88% | &#124; | 109.035 | 61.91 | 11.85% |
| Intel MKL (Seq) | 4.020 | 62.19 | 13.35% | &#124; | 32.076 | 62.35 | 12.02% | &#124; | 108.020 | 62.49 | 11.97% |
| **OpenMP Implementations** | | | | &#124; | | | | &#124; | | | |
| Original (T=24) | 5.360 | 46.65 | 10.02% | &#124; | 43.087 | 46.42 | 8.95% | &#124; | 149.783 | 45.07 | 8.63% |
| Blas-Like (T=16, default) | 0.837 | 298.62 | 64.11% | &#124; | 5.789 | 345.47 | 66.62% | &#124; | 16.752 | 402.93 | 77.15% |
| Blas-Like (T=16, dynamic, 1) | 0.804 | 310.97 | 66.76% | &#124; | 4.977 | 401.82 | 77.49% | &#124; | 15.571 | 433.49 | 83.01% |
| Blas-Like (Optimal) | 0.882 | 283.32 | 60.83% | &#124; | 4.906 | 407.67 | 78.61% | &#124; | 15.571 | 433.50 | 83.01% |
| OpenBLAS (T=16) | 0.662 | 377.37 | 81.02% | &#124; | 4.677 | 427.65 | 82.47% | &#124; | 16.997 | 397.13 | 76.04% |
| **Intel MKL (T=16)** | **0.537** | **465.77** | **100.00%** | &#124; | **3.857** | **518.57** | **100.00%** | &#124; | **12.925** | **522.24** | **100.00%** |
| **CUDA Implementations** | | | | &#124; | | | | &#124; | | | |
| Kernel 1 (16x16) | 1.600 | 156.25 | 33.55% | &#124; | 11.034 | 181.26 | 34.95% | &#124; | 42.411 | 159.16 | 30.48% |
| Kernel 2 (32x32) | 1.411 | 177.18 | 38.04% | &#124; | 9.973 | 200.54 | 38.67% | &#124; | 33.582 | 201.00 | 38.49% |
| cuBLAS | 1.326 | 188.54 | 40.48% | &#124; | 8.921 | 224.19 | 43.23% | &#124; | 29.304 | 230.34 | 44.11% |
| **MPI Implementations** | | | | &#124; | | | | &#124; | | | |
| MPI Original (p=24) | 1.327 | 188.38 | 40.44% | &#124; | 5.558 | 359.82 | 69.39% | &#124; | 19.465 | 346.78 | 66.40% |
| MPI Intel MKL | ~0.450 | 555.56 | 119.28% | &#124; | ~4.500 | 444.44 | 85.70% | &#124; | ~15.000 | 450.00 | 86.17% |