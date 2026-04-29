test_run.py

Purpose
-------
Run profiling tools (Intel Advisor or Valgrind) on compiled builds and save the generated reports to the `experiments/` directory.

Usage
-----
- Run Intel Advisor roofline collection and generate an HTML report:
  python test_run.py --intel-advisor --build <build_dir> --experiment <name> -- [exe args...]

- Run Valgrind cachegrind:
  python test_run.py --valgrind --build <build_dir> --experiment <name> -- [exe args...]

Options
-------
- `--intel-advisor`: Run Intel Advisor roofline/tripcounts/survey (CPU profiling). Generates `experiments/<name>.html`.
- `--valgrind`: Run Valgrind cachegrind. Produces `experiments/<name>.valgrind.txt`.
- `--build` (required): Build directory containing `executable`.
- `--experiment` (required): Base name for output files in `experiments/`.

Behavior
--------
- The script checks if output files already exist and aborts to avoid overwriting.
- For Intel Advisor it attempts to source `/opt/intel/oneapi/setvars.sh` if available to set up the environment.
- For Advisor it creates a temporary project directory for collection, runs the collects and then generates an interactive HTML report saved under `experiments/`. The temporary project directory is removed afterward.
- For Valgrind the script runs `valgrind --tool=cachegrind` and writes the cachegrind log to `experiments/<name>.valgrind.txt`.

Notes
-----
- Intel Advisor and Valgrind must be installed and available on PATH for the respective options to succeed.
- Advisor invocation in this script forces CPU profiling (`--no-profile-gpu`).
