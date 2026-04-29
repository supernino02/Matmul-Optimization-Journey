"""Shared constants and paths for the build/test tools."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # project root (one level up from TOOLS/)
SRC_DIR = ROOT / "src"
VERSIONS_DIR = SRC_DIR / "versions"
CUDA_DIR = SRC_DIR / "cuda"
CUDA_KERNELS_DIR = CUDA_DIR / "kernels"
BUILDS_DIR = ROOT / "builds"
RESULTS_DIR = ROOT / "experiments"
PLOTS_DIR = ROOT / "plots"
SETVARS = "/opt/intel/oneapi/setvars.sh"


COMMON_SRCS = [
    SRC_DIR / "main.c",
    SRC_DIR / "arg_parser.c",
    SRC_DIR / "matrix_utils.c",
    SRC_DIR / "io_utils.c",
]

# Shared flags (safe for all compilers)
DEFAULT_CFLAGS = [
    #"-pg",           # debug symbols (required for profiling tools like Intel Advisor,but remove optimizations)
    "-fopenmp",
    "-Wall",
    "-Wextra"
]

# Extra flags applied only when using ICC (classic Intel compiler)
ICC_EXTRA_FLAGS = ["-diag-disable=10441"]

# Extra flags applied only when using ICX (Intel LLVM-based compiler)
ICX_EXTRA_FLAGS = []

# Extra flags applied only when using GCC
GCC_EXTRA_FLAGS = []
