#!/usr/bin/env python3
"""Merge all experiment CSVs (except blas_hyperparam_tuning) into MERGE_ALL.csv,
then invoke plot_comparison.py with human-readable label names."""

import csv
import glob
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = ROOT / "experiments"
PLOTS_DIR = ROOT / "plots"
MERGE_OUTPUT = EXPERIMENTS_DIR / "MERGE_ALL.csv"
PLOT_SCRIPT = PLOTS_DIR / "plot_comparison.py"
PLOT_OUTPUT = PLOTS_DIR / "MERGE_ALL.png"

EXCLUDE_FILES = {"blas_hyperparam_tuning.csv", "compare_blas_dense.csv", "MERGE_ALL.csv"}

# Human-readable rename mapping
RENAME_MAP = {
    "gcc_0":                "GCC -O0",
    "gcc_1":                "GCC -O1",
    "gcc_2":                "GCC -O2",
    "gcc_3":                "GCC -O3",
    "icc_0":                "ICC -O0",
    "icc_1":                "ICC -O1",
    "icc_2":                "ICC -O2",
    "icc_3":                "ICC -O3",
    "icc_maximum":          "ICC Max",
    "icc_maximum_aligned":  "ICC Max Aligned",
    "icx_0":                "ICX -O0",
    "icx_1":                "ICX -O1",
    "icx_2":                "ICX -O2",
    "icx_3":                "ICX -O3",
    "icx_maximum":          "ICX Max",
    "icx_maximum_aligned":  "ICX Max Aligned",
    "openblas_seq":         "OpenBLAS",
    "mkl_seq":              "MKL",
    "blas_like":            "Our BLAS",
    "const_icc":            "ICC Constants",
    "const_icx":            "ICX Constants",
}

CANONICAL_COLUMNS = ["executable", "n", "real_time_seconds", "cpu_time_seconds"]


def merge_csvs():
    csv_files = sorted(glob.glob(str(EXPERIMENTS_DIR / "*.csv")))
    seen = set()
    rows = []

    for fpath in csv_files:
        fname = os.path.basename(fpath)
        if fname in EXCLUDE_FILES:
            continue
        with open(fpath, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Only keep canonical columns
                try:
                    canonical = (row["executable"], row["n"],
                                 row["real_time_seconds"], row["cpu_time_seconds"])
                except KeyError:
                    continue
                # Filter: only keep rows with n <= 10000
                try:
                    if int(row["n"]) > 10000:
                        continue
                except (ValueError, KeyError):
                    continue
                if canonical in seen:
                    continue
                seen.add(canonical)
                rows.append({k: row[k] for k in CANONICAL_COLUMNS})

    with open(MERGE_OUTPUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CANONICAL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Merged {len(rows)} unique rows from {len(csv_files) - len([x for x in csv_files if os.path.basename(x) in EXCLUDE_FILES])} CSV files → {MERGE_OUTPUT}")
    return len(rows)


def build_plot_command():
    # Build --rename-label pairs
    rename_args = []
    for old, new in RENAME_MAP.items():
        rename_args.extend([old, new])

    cmd = [
        sys.executable,
        str(PLOT_SCRIPT),
        "--experiment", "MERGE_ALL",
        "--x", "n",
        "--y", "real_time_seconds",
        "--label", "executable",

        "--rename-label", *rename_args,
        "--out", str(PLOT_OUTPUT),
    ]
    return cmd


def main():
    # Step 1: Merge
    n = merge_csvs()
    if n == 0:
        print("[ERROR] No rows merged, aborting plot.")
        sys.exit(1)

    # Step 2: Remove existing plot file if it exists (plot_comparison.py refuses to overwrite)
    if PLOT_OUTPUT.exists():
        PLOT_OUTPUT.unlink()
        print(f"[INFO] Removed existing {PLOT_OUTPUT}")

    # Step 3: Plot
    cmd = build_plot_command()
    print(f"[INFO] Running: {' '.join(cmd[:6])} ... --rename-label <{len(RENAME_MAP)} pairs> --out {PLOT_OUTPUT.name}")
    result = subprocess.run(cmd, cwd=str(ROOT))
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
