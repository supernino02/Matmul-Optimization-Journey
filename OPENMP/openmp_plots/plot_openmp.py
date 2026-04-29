#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot OpenMP experiment CSVs with 4 subplots: real_time, cpu_time, speed-up, efficiency.

Usage examples:
  python plot_openmp.py --experiment orig_parallel_threads --exec "blas-like"
  python plot_openmp.py --experiment orig_parallel_threads --exec "blas-like" --x threads --label executable --out openmp.png
  python plot_openmp.py --experiment orig_parallel_threads --exec "blas-like" --log-x --out openmp_log.png

The script groups by label and x column, computes mean/min/max for real_time_seconds
and cpu_time_seconds, derives speed-up and efficiency from the real_time_seconds
(speed-up = T(1)/T(p), efficiency = speed-up/p), and produces a 2x2 figure.

Mandatory flags:
  --experiment EXPERIMENT  Experiment name (file in experiments/<name>.csv)
  --exec EXEC              Base label name (e.g. 'blas-like'). Will be suffixed with 
                           'multithread', 'sequential', and 'ideal'.

Optional flags:
  --x COLUMN               Column to use as x-axis (default: threads)
  --label COLUMN           Column for series labels (default: "executable")
  --seq                    If passed, use the 1-thread execution time as the sequential 
                           baseline to draw the ideal reference curves.
  --out FILENAME           Save plot to this filename (png/pdf); otherwise show interactively
  --log-x                  Use logarithmic scale for the x-axis.
  --log-y                  Use logarithmic scale for all y-axes.
  --best                   Draw a red 'x' on the plot at the overall best x value for 
                           the main executable.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import matplotlib.ticker as ticker  # type: ignore
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "experiments"
PLOTS_DIR = ROOT / "plots"

# -- Columns used ------------------------------------------------------------
REAL_TIME = "real_time_seconds"
CPU_TIME = "cpu_time_seconds"


def prepare_x_for_plot(xseries):
    """Return (xpos_array, xtick_labels_or_None)."""
    xnum = pd.to_numeric(xseries, errors="coerce")
    if xnum.notna().all():
        return xnum.values, None
    uniques = list(dict.fromkeys(xseries.astype(str).tolist()))
    mapping = {v: i for i, v in enumerate(uniques)}
    xpos = xseries.astype(str).map(mapping).values
    return xpos, uniques


def add_speedup_efficiency(grp, xcol, label_col, t1_global):
    """Add speed-up and efficiency columns to a grouped DataFrame using a global baseline."""
    records = []
    for lbl, sub in grp.groupby(label_col):
        for _, row in sub.iterrows():
            p = float(row[xcol])
            su = t1_global / row["real_mean"] if row["real_mean"] > 0 else np.nan
            eff = su / p if p > 0 else np.nan
            records.append({
                label_col: row[label_col],
                xcol: row[xcol],
                "real_mean": row["real_mean"],
                "real_min": row["real_min"],
                "real_max": row["real_max"],
                "cpu_mean": row["cpu_mean"],
                "cpu_min": row["cpu_min"],
                "cpu_max": row["cpu_max"],
                "speedup": su,
                "efficiency": eff,
            })
    return pd.DataFrame(records)


def main():
    p = argparse.ArgumentParser(description="Plot OpenMP experiments (4 subplots)")
    p.add_argument("--experiment", required=True,
                   help="Experiment name (file in experiments/<name>.csv)")
    p.add_argument("--exec", required=True,
                   help="Base label name (e.g. 'blas-like'). Will be suffixed with 'multithread', 'sequential', and 'ideal'.")
    p.add_argument("--x", default="threads",
                   help="Column to use as x-axis (default: threads)")
    p.add_argument("--label", default="executable",
                   help='Column for series labels (default: "executable")')
    p.add_argument("--seq", action="store_true",
                   help="If provided, use the 1-thread time as the baseline for ideal curves.")
    p.add_argument("--out",
                   help="Save plot to this filename (png/pdf); otherwise show interactively")
    p.add_argument("--log-x", action="store_true",
                   help="Use logarithmic scale for the x-axis")
    p.add_argument("--log-y", action="store_true",
                   help="Use logarithmic scale for all y-axes")
    p.add_argument("--best", action="store_true",
                   help="Draw a red 'x' at the overall best x value for the main executable")

    args = p.parse_args()

    # -- Load main CSV ------------------------------------------------------
    csv_path = RESULTS_DIR / f"{args.experiment}.csv"
    if not csv_path.exists():
        print(f"[ERROR] Experiment file not found: {csv_path}")
        sys.exit(2)

    try:
        df = pd.read_csv(csv_path, comment="/")
    except Exception as e:
        print(f"[ERROR] Could not read CSV: {e}")
        sys.exit(3)

    if df.empty:
        print(f"[ERROR] CSV is empty: {csv_path}")
        sys.exit(4)

    xcol = args.x
    for col in [args.label, xcol, REAL_TIME, CPU_TIME]:
        if col not in df.columns:
            print(f"[ERROR] Column '{col}' not found. Available: {', '.join(df.columns)}")
            sys.exit(5)

    # Extract dynamic 'n' value for title if it exists
    n_title_info = ""
    if "n" in df.columns:
        n_vals = df["n"].dropna().unique()
        if len(n_vals) == 1:
            n_title_info = f" (n={n_vals[0]})"

    # Ensure numeric
    df[REAL_TIME] = pd.to_numeric(df[REAL_TIME], errors="coerce")
    df[CPU_TIME] = pd.to_numeric(df[CPU_TIME], errors="coerce")
    df[xcol] = pd.to_numeric(df[xcol], errors="coerce")
    
    # Drop NAs for numerical metrics, but preserve rows missing just the label by filling
    df = df.dropna(subset=[REAL_TIME, CPU_TIME, xcol])
    df[args.label] = df[args.label].fillna("sequential")

    if args.log_x:
        df = df[df[xcol] > 0]
    if args.log_y:
        df = df[(df[REAL_TIME] > 0) & (df[CPU_TIME] > 0)]

    if df.empty:
        print("[ERROR] No valid rows remaining after filtering")
        sys.exit(6)

    # -- Compute Global Baseline T(1) ----------------------------------------
    seq_runs = df[df[xcol] == 1]
    if not seq_runs.empty:
        # Absolute best sequential time if explicit p=1 is available
        t1_global = seq_runs[REAL_TIME].min()
    else:
        # Extrapolate theoretical T(1) if data starts at p > 1
        min_x = df[xcol].min()
        best_at_min_x = df[df[xcol] == min_x][REAL_TIME].min()
        t1_global = best_at_min_x * min_x

    # -- Aggregate -----------------------------------------------------------
    agg = (
        df.groupby([args.label, xcol])
        .agg({REAL_TIME: ["mean", "min", "max"], CPU_TIME: ["mean", "min", "max"]})
    )
    agg.columns = ["real_mean", "real_min", "real_max", "cpu_mean", "cpu_min", "cpu_max"]
    agg = agg.reset_index()

    grp = add_speedup_efficiency(agg, xcol, args.label, t1_global)
    
    label_values = sorted(grp[args.label].unique(), key=str)

    # -- Compute global best -------------------------------------------------
    global_best_x = None
    global_best_real = None
    if args.best and not grp.empty:
        main_lbl = label_values[0]
        g_main = grp[grp[args.label] == main_lbl]
        if not g_main.empty:
            per_x_real = g_main.groupby(xcol)["real_mean"].mean()
            global_best_x = per_x_real.idxmin()
            global_best_real = per_x_real[global_best_x]

    # -- Subplot configuration -----------------------------------------------
    metrics = [
        ("real_mean", "real_min", "real_max", "Real Time (s)"),
        ("cpu_mean",  "cpu_min",  "cpu_max",  "CPU Time (s)"),
        ("speedup",   None,       None,       "Speed-up"),
        ("efficiency", None,      None,       "Efficiency"),
    ]

    # Initialize figure
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
    axes = axes.flatten()

    # Link the calculated global baseline to the ideal curves if requested
    T1_ideal = None
    if args.seq and not grp.empty:
        T1_ideal = t1_global

    for ax_idx, (mean_col, min_col, max_col, ylabel) in enumerate(metrics):
        ax = axes[ax_idx]

        last_xtick_labels = None
        for s_idx, lbl in enumerate(label_values):
            g = grp[grp[args.label] == lbl].copy()
            if g.empty: continue
            
            xpos, xtick_labels = prepare_x_for_plot(g[xcol])
            order = np.argsort(xpos)
            xplot = xpos[order]
            mean = g[mean_col].values[order]
            
            # --- Dynamically map OpenMP schedules if requested ---
            if args.label == "schedule":
                sched_map = {
                    "1": "static", "1.0": "static",
                    "2": "dynamic", "2.0": "dynamic",
                    "3": "guided", "3.0": "guided",
                    "4": "auto", "4.0": "auto"
                }
                display_label = sched_map.get(str(lbl), str(lbl))
            elif s_idx == 0:
                display_label = f"{args.exec} multithread"
            else:
                display_label = str(lbl)
                
            # Plot without explicit markers (relies on automatic color cycler)
            ax.plot(xplot, mean, label=display_label, linewidth=2.0)

            if min_col and max_col:
                ymin_vals = g[min_col].values[order]
                ymax_vals = g[max_col].values[order]
                ax.fill_between(xplot, ymin_vals, ymax_vals, alpha=0.15)

            if xtick_labels is not None:
                last_xtick_labels = xtick_labels

        # -- ideal reference lines (grey dotted) across all 4 plots ----------
        ideal_label = f"{args.exec} ideal"
        all_x = sorted(grp[xcol].dropna().unique())
        
        if mean_col == "speedup":
            ax.plot(all_x, all_x, linestyle=":", linewidth=1.5, color="gray",
                    alpha=0.7, label=ideal_label)
        elif mean_col == "efficiency":
            ax.axhline(y=1.0, linestyle=":", linewidth=1.5, color="gray",
                       alpha=0.7, label=ideal_label)
        elif mean_col == "real_mean" and T1_ideal is not None:
            ideal_y = [T1_ideal / float(x) for x in all_x]
            ax.plot(all_x, ideal_y, linestyle=":", linewidth=1.5, color="gray",
                    alpha=0.7, label=ideal_label)
        elif mean_col == "cpu_mean" and T1_ideal is not None:
            ax.axhline(y=T1_ideal, linestyle=":", linewidth=1.5, color="gray",
                       alpha=0.7, label=ideal_label)

        # -- best point (--best) ON ALL PLOTS using global best X -----------
        if args.best and global_best_x is not None:
            b_x = float(global_best_x)
            t_str = "thread" if b_x == 1 else "threads"
            
            if ax_idx == 0:
                best_label = f"Optimal execution\n({global_best_real:.4g}s at {b_x:g} {t_str})"
            else:
                best_label = f"Optimal execution\n(at {b_x:g} {t_str})"
            
            main_lbl = label_values[0]
            g_best = grp[(grp[args.label] == main_lbl) & (grp[xcol] == b_x)]
            if not g_best.empty:
                b_y = g_best[mean_col].values[0]
                ax.plot(b_x, b_y, marker='x', color='red', markersize=8, markeredgewidth=2,
                        linestyle='None', label=best_label, zorder=10)

        if last_xtick_labels is not None:
            ax.set_xticks(range(len(last_xtick_labels)))
            ax.set_xticklabels(last_xtick_labels, rotation=45)

        ax.set_title(ylabel)
        ax.grid(False)

        if args.log_y:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: "%g" % y))

        if args.log_x:
            ax.set_xscale("log")
            try:
                numeric_x = pd.to_numeric(grp[xcol], errors="coerce").dropna().unique()
            except Exception:
                numeric_x = np.array([])
            if numeric_x.size > 0:
                xticks = np.sort(numeric_x.astype(float))
                ax.xaxis.set_major_locator(ticker.FixedLocator(xticks))
                ax.xaxis.set_minor_locator(ticker.NullLocator())
                ax.set_xticks(xticks)
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: "%g" % x))

    for ax in axes[2:]:
        ax.set_xlabel(xcol.replace("_", " ").title())

    fig.suptitle(f"OpenMP Scaling Analysis{n_title_info}", fontsize=15, fontweight="bold")
    
# Restrict the subplots to leave room at the bottom for the horizontal legend
    fig.tight_layout(rect=[0, 0.07, 1, 0.96])
    
    # Extract handles from the first subplot to generate the unified Figure legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=len(labels))
    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_file = PLOTS_DIR / out_path
        else:
            out_file = out_path
        out_file.parent.mkdir(parents=True, exist_ok=True)
        if out_file.exists():
            print(f"[ERROR] Output file already exists: {out_file}")
            sys.exit(9)
        plt.savefig(out_file, dpi=150)
        print(f"[OK] Plot saved to {out_file}")
    else:
        plt.show()

if __name__ == "__main__":
    main()