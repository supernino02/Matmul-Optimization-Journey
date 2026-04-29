#!/usr/bin/env python3
# filepath: /home/stud/S5312342/Downloads/HPC-main/plot.py
"""Plot experiment CSVs from the experiments/ directory.

Usage examples:
  python plot.py --experiment align_vs_base_2 --x n --y real_time_seconds --label executable
  python plot.py --experiment align_vs_base_2 --x n --y cpu_time_seconds --label executable --out out.png
  python plot.py --experiment compare_0 --compare-to all_comparisons_5000 gcc_0 --rename-label gcc_0 "GCC -O0" gcc_1 "GCC -O1"

The script groups by the label and x column, computes mean/min/max of y, plots the mean
as a line and a shaded band between min and max.

Optional flags:
  --compare-to EXPERIMENT LABEL   Load a second experiment CSV and overlay one
                                  label from it as a dotted gray reference line.
  --rename-label K V [K V ...]    Rename labels in the legend (pairs of old new).
"""
import argparse
import sys
from pathlib import Path

import pandas as pd  # type: ignore
import matplotlib.pyplot as plt # type: ignore
import matplotlib.ticker as ticker # type: ignore
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "experiments"
PLOTS_DIR = ROOT / "plots"

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
    p.add_argument("--x", default="n",help="Column to use as x-axis")
    p.add_argument("--y", default="real_time_seconds", help="Column to use as y-axis")
    p.add_argument("--label", default="executable", help='Column to use for series labels (default: "executable")')
    p.add_argument("--out", help="If provided, save plot to this filename (png/pdf), otherwise show interactively")
    p.add_argument("--log-x", action="store_true", help="Use logarithmic scale for the x-axis (drops non-positive x values)")
    p.add_argument("--log-y", action="store_true", help="Use logarithmic scale for the y-axis (drops non-positive y values)")
    p.add_argument("--compare-to", nargs=2, metavar=("EXPERIMENT", "LABEL"),
                   help="Overlay a single label from another experiment as a dotted gray reference line")
    p.add_argument("--rename-label", nargs="+", metavar="K_V",
                   help="Pairs of old_label new_label to rename labels in the legend")

    args = p.parse_args()

    # Build rename mapping from --rename-label pairs
    rename_map = {}
    if args.rename_label:
        rl = args.rename_label
        if len(rl) % 2 != 0:
            print("[ERROR] --rename-label requires an even number of arguments (pairs of old new)")
            sys.exit(11)
        for i in range(0, len(rl), 2):
            rename_map[rl[i]] = rl[i + 1]

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

    # If log scale requested, remove non-positive y values (log scale can't handle <= 0)
    if args.log_y:
        nonpos = df[ycol] <= 0
        if nonpos.any():
            print(f"[WARNING] Dropping {nonpos.sum()} rows with non-positive y-values because log scale requested")
            df = df[df[ycol] > 0]
        if df.empty:
            print(f"[ERROR] No positive y values available for log scale")
            sys.exit(10)

    if args.log_x:
        nonpos = df[xcol] <= 0
        if nonpos.any():
            print(f"[WARNING] Dropping {nonpos.sum()} rows with non-positive x-values because log scale requested")
            df = df[df[xcol] > 0]
        if df.empty:
            print(f"[ERROR] No positive x values available for log scale")
            sys.exit(10)

    # ── Load --compare-to data (if provided) ────────────────────────────────
    compare_grp = None
    compare_label_name = None
    if args.compare_to:
        cmp_exp, cmp_lbl = args.compare_to
        cmp_csv = RESULTS_DIR / f"{cmp_exp}.csv"
        if not cmp_csv.exists():
            print(f"[ERROR] Compare-to experiment file not found: {cmp_csv}")
            sys.exit(2)
        try:
            cmp_df = pd.read_csv(cmp_csv)
        except Exception as e:
            print(f"[ERROR] Could not read compare-to CSV: {e}")
            sys.exit(3)
        if args.label not in cmp_df.columns:
            print(f"[ERROR] Label column '{args.label}' not found in compare-to CSV.")
            sys.exit(5)
        if xcol not in cmp_df.columns or ycol not in cmp_df.columns:
            print(f"[ERROR] x or y column not found in compare-to CSV.")
            sys.exit(7)
        cmp_df = cmp_df[cmp_df[args.label] == cmp_lbl].copy()
        if cmp_df.empty:
            print(f"[ERROR] Label '{cmp_lbl}' not found in compare-to experiment '{cmp_exp}'.")
            sys.exit(12)
        cmp_df[ycol] = pd.to_numeric(cmp_df[ycol], errors="coerce")
        cmp_df = cmp_df[cmp_df[ycol].notna()]
        if args.log_y:
            cmp_df = cmp_df[cmp_df[ycol] > 0]
        if args.log_x:
            cmp_df = cmp_df[cmp_df[xcol] > 0]
        compare_grp = cmp_df.groupby(xcol)[ycol].agg(["mean", "min", "max"]).reset_index()
        compare_label_name = rename_map.get(cmp_lbl, cmp_lbl)

    # group by label and x, compute mean/min/max
    grp = df.groupby([args.label, xcol])[ycol].agg(["mean", "min", "max"]).reset_index()

    plt.figure(figsize=(10, 6))

    label_values = grp[args.label].unique()
    # define a small set of distinct styles to make series visually different
    styles = [
        {"linestyle": "-",  "linewidth": 2.0, "marker": "o"},
        {"linestyle": "-", "linewidth": 2.0, "marker": "s"},
        {"linestyle": "-", "linewidth": 2.0, "marker": "^"},
        {"linestyle": "-",  "linewidth": 2.0, "marker": "D"},
    ]

    last_xtick_labels = None
    for idx, lbl in enumerate(sorted(label_values, key=lambda s: str(s))):
        g = grp[grp[args.label] == lbl].copy()
        # prepare x for plotting (numeric or categorical mapping)
        xpos, xtick_labels = prepare_x_for_plot(g[xcol])
        # sort by xpos so lines are ordered
        order = np.argsort(xpos)
        xplot = xpos[order]
        mean = g["mean"].values[order]
        ymin = g["min"].values[order]
        ymax = g["max"].values[order]

        style = styles[idx % len(styles)]
        display_label = rename_map.get(str(lbl), str(lbl))
        plt.plot(
            xplot,
            mean,
            label=display_label,
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            marker=style["marker"],
        )
        # slightly less opaque band so thicker lines stand out
        plt.fill_between(xplot, ymin, ymax, alpha=0.15)

        # remember xtick labels if categorical mapping was used
        if xtick_labels is not None:
            last_xtick_labels = xtick_labels

    # ── Plot --compare-to reference line (dotted gray) ─────────────────────
    if compare_grp is not None and not compare_grp.empty:
        cxpos, cxtick_labels = prepare_x_for_plot(compare_grp[xcol])
        corder = np.argsort(cxpos)
        cxplot = cxpos[corder]
        cmean = compare_grp["mean"].values[corder]
        cymin = compare_grp["min"].values[corder]
        cymax = compare_grp["max"].values[corder]
        plt.plot(
            cxplot, cmean,
            label=compare_label_name,
            linestyle=":", linewidth=2.0, color="gray", marker="x",
        )
        plt.fill_between(cxplot, cymin, cymax, alpha=0.10, color="gray")
        if cxtick_labels is not None:
            last_xtick_labels = cxtick_labels

    # if any series required categorical x labels, set them once
    if last_xtick_labels is not None:
        plt.xticks(ticks=range(len(last_xtick_labels)), labels=last_xtick_labels, rotation=45)

    # Reorder legend so the --compare-to reference line appears first
    handles, labels = plt.gca().get_legend_handles_labels()
    if compare_label_name is not None and compare_label_name in labels:
        ci = labels.index(compare_label_name)
        handles = [handles[ci]] + handles[:ci] + handles[ci+1:]
        labels = [labels[ci]] + labels[:ci] + labels[ci+1:]
    plt.legend(handles, labels, title=args.label)
    plt.xlabel("Matrix Dimension")
    plt.ylabel("Executions Seconds")
    # remove background grid as requested
    plt.grid(False)
    # add title
    plt.title("Execution Time Comparison")

    # apply logarithmic scale to y-axis if requested
    if args.log_y:
        plt.yscale("log")
        # use a numeric formatter for log ticks instead of mathtext (10^x)
        ax = plt.gca()
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: ('%g' % y)))

    # apply logarithmic scale to x-axis if requested
    if args.log_x:
        plt.xscale("log")
        # use a numeric formatter for log ticks instead of mathtext (10^x)
        ax = plt.gca()
        # compute numeric x values present in the grouped data and use them as ticks
        try:
            numeric_x = pd.to_numeric(grp[xcol], errors="coerce").dropna().unique()
        except Exception:
            numeric_x = np.array([])
        if numeric_x.size > 0:
            xticks = np.sort(numeric_x.astype(float))
            # force major ticks only at the data x positions and remove minor ticks
            ax.xaxis.set_major_locator(ticker.FixedLocator(xticks))
            ax.xaxis.set_minor_locator(ticker.NullLocator())
            ax.set_xticks(xticks)
            # add multiplicative 10% padding on log axis
            lower = xticks.min()
            upper = xticks.max()
            if lower > 0 and upper > 0:
                pad_factor = 1.1
                ax.set_xlim(lower / pad_factor, upper * pad_factor)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: ('%g' % x)))
    else:
        # for linear x-axis add 10% additive padding so points aren't at the very edges
        ax = plt.gca()
        try:
            xmin, xmax = ax.get_xlim()
            rng = xmax - xmin
            if rng == 0:
                # single-value case: expand by ±10%
                pad = abs(xmin) * 0.1 if xmin != 0 else 1.0
            else:
                pad = rng * 0.1
            ax.set_xlim(xmin - pad, xmax + pad)
        except Exception:
            pass
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
