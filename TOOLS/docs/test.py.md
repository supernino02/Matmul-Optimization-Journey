test.py

Purpose
-------
Run compiled builds with parameter sweeps and collect outputs into a CSV stored in `experiments/`. Useful for benchmarking multiple builds across input sizes and other runtime parameters.

Usage
-----
- Minimum: python test.py --builds <build1> <build2> --iterations 1 --experiment <name> -- [runtime args...]

Example:
- python test.py --builds gcc_1 icx_1 --iterations 3 --experiment compare_1 -- -n 512 -verify
  (Note: arguments after `--` are forwarded to the executables)

Important options
-----------------
- `--builds` (required): List of build directories in `builds/` to run (e.g., `gcc_1`).
- `--iterations`: Number of times to repeat each run (default: 1).
- `--verbose`: 0/1/2 verbosity levels.
- `--experiment`: Name of the output CSV in `experiments/<name>.csv`. If omitted the script runs in dry-run mode (no CSV written, verbose=2).
- `--build-info`: Include build metadata fields from `builds/builds.csv` in the output CSV.

How delegated args work
-----------------------
Arguments not recognized by the harness are forwarded to the executable. Options with multiple values are swept over, producing combinations. Single-dash tokens are treated as boolean flags.

Output format
-------------
- The executable is expected to print key=value lines to stdout (e.g., `real_time_seconds=1.234`). The script parses these lines into CSV columns.
- The CSV contains columns: `executable`, `returncode`, optional build metadata columns, delegated argument columns, and parsed output keys.

Notes
-----
- The script regenerates `builds/builds.csv` by invoking `build.py --list-builds` before running.
- If the experiment CSV already exists the script aborts to avoid overwriting.
