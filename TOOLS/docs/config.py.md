config.py

Purpose
-------
Shared constants and paths used by the helper tools in `TOOLS/` (build/test/plot). Centralizes project directories and compiler flags.

How to use
----------
This file is imported by other scripts. Typical usage examples:
- from config import BUILDS_DIR, SRC_DIR, VERSIONS_DIR
- Use `DEFAULT_CFLAGS`, `GCC_EXTRA_FLAGS`, `ICC_EXTRA_FLAGS` when assembling compile commands.

Key constants
-------------
- `ROOT`            : Project root directory
- `SRC_DIR`         : `src/` directory
- `VERSIONS_DIR`    : `src/versions` (C sources for different implementations)
- `CUDA_DIR`        : `src/cuda`
- `CUDA_KERNELS_DIR`: `src/cuda/kernels`
- `BUILDS_DIR`      : `builds/` directory where compiled builds are stored
- `RESULTS_DIR`     : `experiments/` directory for CSV/HTML outputs
- `PLOTS_DIR`       : `plots/` directory for generated plots
- `SETVARS`         : Path to Intel oneAPI setvars script

Flags
-----
- `COMMON_SRCS`     : List of common source files included in most builds
- `DEFAULT_CFLAGS`  : Flags applied to all compilers (e.g. `-fopenmp`, `-Wall`)
- `ICC_EXTRA_FLAGS` : Additional flags for classic Intel compiler
- `ICX_EXTRA_FLAGS` : Additional flags for Intel ICX
- `GCC_EXTRA_FLAGS` : Additional flags for GCC

Output
------
This module only defines constants; it has no direct runtime outputs. Other scripts will read these constants to find files, directories and default flags.
