test_cuda.py

Purpose
-------
A variant of `test.py` specialized for runs of CUDA-enabled builds. It also computes GFLOP/s from the reported runtime when possible and includes it in the output.

Usage
-----
- python test_cuda.py --builds <build1> --experiment <name> -- -n 4096

Options
-------
Same controller options as `test.py`:
- `--builds` (required)
- `--iterations` (default 1)
- `--verbose` (0/1/2)
- `--experiment` (writes `experiments/<name>.csv`)
- `--build-info`

CUDA-specific behavior
----------------------
- After running each executable the script looks for `n` (problem size) among delegated args and `real_time_seconds` among the program output. When both are present and valid numbers it computes GFLOP/s using the formula: GFLOP/s = 2*N^3 / (time_seconds * 1e9) and adds a `gflops` column to the output CSV.

Output
------
- Writes `experiments/<name>.csv` with parsed key=value outputs plus the computed `gflops` where applicable.
