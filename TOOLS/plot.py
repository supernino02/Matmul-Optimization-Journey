#!/usr/bin/env python3
# filepath: /home/stud/S5312342/Downloads/HPC-main/plot.py
"""Plot experiment CSVs from the experiments/ directory.

Usage examples:
  python plot.py --experiment align_vs_base_2 --x n --y real_time_seconds --label executable
  python plot.py --experiment align_vs_base_2 --x n --y cpu_time_seconds --label executable --out out.png

The script groups by the label and x column, computes mean/min/max of y, plots the mean
as a line and a shaded band between min and max.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd  # type: ignore
import matplotlib.pyplot as plt
import numpy as np

from config import RESULTS_DIR,PLOTS_DIR


def choose_default_column(df, prefer: list, exclude: set = None):
    exclude = exclude or set()
    for c in prefer:
        if c in df.columns and c not in exclude:
            return c
    # fallback: first numeric column not excluded
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_integer_dtype(df[c]):
            return c
        # also accept columns that convert to numeric
        try:
            ser = pd.to_numeric(df[c], errors="coerce")
            if ser.notna().any():
                return c
        except Exception:
            pass
    # last resort: any column not excluded
    for c in df.columns:
        if c not in exclude:
            return c
    return None


def prepare_x_for_plot(xseries):
    # try numeric conversion
    xnum = pd.to_numeric(xseries, errors="coerce")
    if xnum.notna().all():
        return xnum.values, None
    # partially numeric: keep numeric where possible, fallback to categorical mapping
    if xnum.notna().any():
        # map distinct original values preserving order of appearance
        uniques = list(dict.fromkeys(xseries.astype(str).tolist()))
        mapping = {v: i for i, v in enumerate(uniques)}
        xpos = xseries.astype(str).map(mapping).values
        return xpos, uniques
    # fully non-numeric: categorical mapping
    uniques = list(dict.fromkeys(xseries.astype(str).tolist()))
    mapping = {v: i for i, v in enumerate(uniques)}
    xpos = xseries.astype(str).map(mapping).values
    return xpos, uniques


def main():
    p = argparse.ArgumentParser(description="Plot experiment CSVs from experiments/")
    p.add_argument("--experiment", required=True, help="Experiment name (file in experiments/<name>.csv)")
    p.add_argument("--x", help="Column to use as x-axis")
    p.add_argument("--y", help="Column to use as y-axis")
    p.add_argument("--label", default="executable", help='Column to use for series labels (default: "executable")')
    p.add_argument("--out", help="If provided, save plot to this filename (png/pdf), otherwise show interactively")
    args = p.parse_args()

    csv_path = RESULTS_DIR / f"{args.experiment}.csv"
    if not csv_path.exists():
        print(f"[ERROR] Experiment file not found: {csv_path}")
        sys.exit(2)

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[ERROR] Could not read CSV: {e}")
        sys.exit(3)

    if df.empty:
        print(f"[ERROR] CSV is empty: {csv_path}")
        sys.exit(4)

    if args.label not in df.columns:
        print(f"[ERROR] Label column '{args.label}' not found in CSV. Available columns: {', '.join(df.columns)}")
        sys.exit(5)

    xcol = args.x
    ycol = args.y
    # sensible defaults
    xcol = xcol or choose_default_column(df, prefer=["n", "size", "x", "N"], exclude={args.label})
    ycol = ycol or choose_default_column(df, prefer=["real_time_seconds", "cpu_time_seconds", "time", "duration"], exclude={args.label, xcol})

    if xcol is None or ycol is None:
        print(f"[ERROR] Could not determine x/y columns. CSV columns: {', '.join(df.columns)}")
        sys.exit(6)

    if xcol not in df.columns or ycol not in df.columns:
        print(f"[ERROR] x or y column not found. x='{xcol}', y='{ycol}'. Available columns: {', '.join(df.columns)}")
        sys.exit(7)

    # Drop rows where y is missing or not numeric
    df_y = pd.to_numeric(df[ycol], errors="coerce")
    if df_y.isna().all():
        print(f"[ERROR] y column '{ycol}' has no numeric values")
        sys.exit(8)
    df = df.copy()
    df[ycol] = df_y
    df = df[df[ycol].notna()]

    # group by label and x, compute mean/min/max
    grp = df.groupby([args.label, xcol])[ycol].agg(["mean", "min", "max"]).reset_index()

    plt.figure(figsize=(10, 6))

    label_values = grp[args.label].unique()
    for lbl in sorted(label_values, key=lambda s: str(s)):
        g = grp[grp[args.label] == lbl].copy()
        # prepare x for plotting (numeric or categorical mapping)
        xpos, xtick_labels = prepare_x_for_plot(g[xcol])
        # sort by xpos so lines are ordered
        order = np.argsort(xpos)
        xplot = xpos[order]
        mean = g["mean"].values[order]
        ymin = g["min"].values[order]
        ymax = g["max"].values[order]

        plt.plot(xplot, mean, label=str(lbl))
        plt.fill_between(xplot, ymin, ymax, alpha=0.25)

        # if categorical mapping used, set xticks
        if xtick_labels is not None:
            plt.xticks(ticks=range(len(xtick_labels)), labels=xtick_labels, rotation=45)

    plt.legend(title=args.label)
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.grid(True)
    plt.tight_layout()

    if args.out:
        # determine destination: if absolute path provided, use it; otherwise place under PLOTS_DIR
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_file = PLOTS_DIR / out_path
        else:
            out_file = out_path

        # ensure parent directory exists
        out_file.parent.mkdir(parents=True, exist_ok=True)

        # safety check: do not overwrite existing file
        if out_file.exists():
            print(f"[ERROR] Output file already exists: {out_file}")
            sys.exit(9)

        plt.savefig(out_file)
        print(f"[OK] Plot saved to {out_file}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
