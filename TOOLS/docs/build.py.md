build.py

Purpose
-------
Compile and manage builds for the HPC matrix-multiply project. Supports native C compilers (gcc, icx, icc) and CUDA (nvcc). Stores compiled outputs and metadata under the project's `builds/` directory.

Basic usage
-----------
- Compile a C version with a chosen compiler:
  python build.py --version <version> --compiler <gcc|icx|icc> [options]

- Compile a CUDA kernel (nvcc):
  python build.py --version <kernel_name> --compiler nvcc [--params "..."]

- Regenerate the builds summary CSV:
  python build.py --list-builds

Key options
-----------
- `--version, -V`  : Version name (e.g. `original`, `mkl`, `openblas`, or CUDA kernel name).
- `--compiler`    : Compiler executable name (`gcc`, `icx`, `icc`, `nvcc`).
- `--name, -n`    : Custom build directory name under `builds/`.
- `--params, -p`  : Extra compiler flags (quoted string). For MKL/OpenBLAS versions you must specify `-DSEQUENTIAL` or `-DPARALLEL`.
- `--report`      : Generate compiler vectorisation/optimization reports (writes `.opt.txt` or vendor reports into the build dir).
- `--native`      : Enable architecture-specific tuning (e.g. `-march=native` or `-xHost`).
- `--unroll`      : Enable aggressive inlining/unrolling flags.
- `--best-flags`  : Shortcut enabling a set of maximum-optimization flags.
- `--list-builds` : Recreate `builds/builds.csv` from found `metadata.json` files and print it.

CUDA specifics
--------------
When `--compiler nvcc` is used, `build.py` compiles the kernel `.cu` and a `cuda_wrapper.cu`, compiles common C sources with `gcc`, then links via `nvcc`. Provide kernel name via `--version` that corresponds to `src/cuda/kernels/<name>.cu`.

Outputs
-------
- A new directory under `builds/` (e.g. `gcc_1/` or custom `my-build/`) containing:
  - `executable`        : The compiled binary
  - `metadata.json`     : JSON describing compiler, flags, version
  - `<executable>.opt.txt` (optional): Compiler report if `--report` used
- `builds/builds.csv` can be regenerated with `--list-builds` and summarizes all builds.

Notes
-----
- For MKL and OpenBLAS versions the script adds include/link flags automatically but requires the threading macro (`-DSEQUENTIAL` or `-DPARALLEL`).
- Duplicate builds (same version, compiler and flags) are skipped.
- For Intel compilers the script tries to source `/opt/intel/oneapi/setvars.sh` if present.
