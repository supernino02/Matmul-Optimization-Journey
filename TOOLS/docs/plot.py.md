plot.py

Purpose
-------
Plot experiment CSV files from the project's `experiments/` directory. The script groups runs by a label column and x column, computes mean/min/max of the y column and renders a line plot with a shaded min-max band.

Usage
-----
Basic usage requires selecting an experiment CSV and the columns to use for x/y and series labels:
- python plot.py --experiment <name> --x <xcol> --y <ycol> [--label <label_col>] [--out out.png]

Example:
- python plot.py --experiment align_vs_base_2 --x n --y real_time_seconds --label executable
- python plot.py --experiment align_vs_base_2 --x n --y real_time_seconds --label executable --out out.png

Defaults and auto-selection
---------------------------
- If `--label` is omitted it defaults to the `executable` column.
- If `--x` or `--y` are omitted the script tries to choose sensible defaults (e.g. `n` for x and `real_time_seconds` or `cpu_time_seconds` for y).
- X values that are not numeric are mapped categorically while preserving order.

Output
------
- If `--out` is provided, the plot is saved (PNG or PDF) under the given path. If a relative filename is provided, it is saved under the project's `plots/` directory.
- If `--out` is omitted, the plot is displayed interactively using matplotlib.
- The script aborts with an error if required columns or the experiment CSV are missing, or if the output file already exists.

Notes
-----
- The script prevents overwriting existing output files.
- Uses Pandas + Matplotlib and requires them in the environment.
