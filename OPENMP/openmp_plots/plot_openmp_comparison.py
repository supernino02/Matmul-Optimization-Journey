#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot OpenMP experiment CSVs with 4 subplots comparing N implementations.

Usage examples:
  python plot_openmp_n.py --experiments openblas_exp myblas_exp mkl_exp --execs "OpenBLAS" "My-BLAS" "Intel MKL"
  python plot_openmp_n.py --experiments exp1 exp2 exp3 --execs "V1" "V2" "V3" --seq --best --out comparison.png

Mandatory flags:
  --experiments EXP1 EXP2 ...  One or more experiment names (files in experiments/<name>.csv)
  --execs EXEC1 EXEC2 ...      One or more base label names corresponding to experiments

Optional flags:
  --x COLUMN               Column to use as x-axis (default: threads)
  --label COLUMN           Column for series labels (default: "executable")
  --seq                    Use the 1-thread execution time as the baseline for ideal curves.
  --out FILENAME           Save plot to this filename (png/pdf); otherwise show interactively
  --log-x                  Use logarithmic scale for the x-axis.
  --log-y                  Use logarithmic scale for all y-axes.
  --best                   Draw a cross marker at the overall best x value for each executable.
"""
import argparse
import sys
import math
from pathlib import Path

import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import matplotlib.ticker as ticker  # type: ignore
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "experiments"
PLOTS_DIR = ROOT / "plots"

REAL_TIME = "real_time_seconds"
CPU_TIME = "cpu_time_seconds"


def prepare_x_for_plot(xseries):
    xnum = pd.to_numeric(xseries, errors="coerce")
    if xnum.notna().all():
        return xnum.values, None
    uniques = list(dict.fromkeys(xseries.astype(str).tolist()))
    mapping = {v: i for i, v in enumerate(uniques)}
    xpos = xseries.astype(str).map(mapping).values
    return xpos, uniques


def add_speedup_efficiency(grp, xcol, label_col, t1_global):
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
    p = argparse.ArgumentParser(description="Plot OpenMP experiments Comparison (N files)")
    p.add_argument("--experiments", nargs='+', required=True,
                   help="One or more experiment names (files in experiments/<name>.csv)")
    p.add_argument("--execs", nargs='+', required=True,
                   help="One or more base label names (e.g. 'OpenBLAS' 'My-BLAS').")
    p.add_argument("--x", default="threads", help="Column to use as x-axis")
    p.add_argument("--label", default="executable", help='Column for series labels')
    p.add_argument("--seq", action="store_true", help="Use 1-thread time as baseline for ideal curves.")
    p.add_argument("--out", help="Save plot to this filename")
    p.add_argument("--log-x", action="store_true", help="Log scale for x-axis")
    p.add_argument("--log-y", action="store_true", help="Log scale for y-axes")
    p.add_argument("--best", action="store_true", help="Draw cross at best x value")
    p.add_argument("--xlim", nargs=2, type=float, metavar=('MIN', 'MAX'), 
                   help="Limit the x-axis to the specified range (e.g., --xlim 1 16)")

    args = p.parse_args()

    # Step 1: Validation
    if len(args.experiments) != len(args.execs):
        print(f"[ERROR] Mismatch: Provided {len(args.experiments)} experiments but {len(args.execs)} executables.")
        sys.exit(1)

    xcol = args.x
    datasets = []
    n_titles = set()

    # Process all CSVs
    for exp_name, exec_name in zip(args.experiments, args.execs):
        csv_path = RESULTS_DIR / f"{exp_name}.csv"
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

        for col in [args.label, xcol, REAL_TIME, CPU_TIME]:
            if col not in df.columns:
                print(f"[ERROR] Column '{col}' not found in {exp_name}.")
                sys.exit(5)

        if "n" in df.columns:
            n_vals = df["n"].dropna().unique()
            if len(n_vals) == 1:
                n_titles.add(str(n_vals[0]))

        df[REAL_TIME] = pd.to_numeric(df[REAL_TIME], errors="coerce")
        df[CPU_TIME] = pd.to_numeric(df[CPU_TIME], errors="coerce")
        df[xcol] = pd.to_numeric(df[xcol], errors="coerce")
        
        df = df.dropna(subset=[REAL_TIME, CPU_TIME, xcol])
        df[args.label] = df[args.label].fillna("sequential")
        

        if args.log_x: df = df[df[xcol] > 0]
        if args.log_y: df = df[(df[REAL_TIME] > 0) & (df[CPU_TIME] > 0)]

        if df.empty:
            print(f"[ERROR] No valid rows remaining in {exp_name} after filtering")
            sys.exit(6)

        seq_runs = df[df[xcol] == 1]
        if not seq_runs.empty:
            t1_global = seq_runs[REAL_TIME].min()
        else:
            min_x = df[xcol].min()
            best_at_min_x = df[df[xcol] == min_x][REAL_TIME].min()
            t1_global = best_at_min_x * min_x

        agg = (
            df.groupby([args.label, xcol])
            .agg({REAL_TIME: ["mean", "min", "max"], CPU_TIME: ["mean", "min", "max"]})
        )
        agg.columns = ["real_mean", "real_min", "real_max", "cpu_mean", "cpu_min", "cpu_max"]
        agg = agg.reset_index()

        grp = add_speedup_efficiency(agg, xcol, args.label, t1_global)
        label_values = sorted(grp[args.label].unique(), key=str)

        global_best_x = None
        global_best_real = None
        if args.best and not grp.empty:
            main_lbl = label_values[0]
            g_main = grp[grp[args.label] == main_lbl]
            if not g_main.empty:
                per_x_real = g_main.groupby(xcol)["real_mean"].mean()
                global_best_x = per_x_real.idxmin()
                global_best_real = per_x_real[global_best_x]

        datasets.append({
            "grp": grp,
            "t1_global": t1_global,
            "global_best_x": global_best_x,
            "global_best_real": global_best_real,
            "exec_name": exec_name,
            "label_values": label_values
        })

    metrics = [
        ("real_mean", "real_min", "real_max", "Real Time (s)"),
        ("cpu_mean",  "cpu_min",  "cpu_max",  "CPU Time (s)"),
        ("speedup",   None,       None,       "Speed-up"),
        ("efficiency", None,      None,       "Efficiency"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    axes = axes.flatten()
    
    # Step 2: Dynamic Color Allocation
    color_palette = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for ax_idx, (mean_col, min_col, max_col, ylabel) in enumerate(metrics):
        ax = axes[ax_idx]
        last_xtick_labels = None

        for ds_idx, ds in enumerate(datasets):
            color = color_palette[ds_idx % len(color_palette)]
            exec_name = ds["exec_name"]
            grp = ds["grp"]
            
            main_lbl = ds["label_values"][0]
            g = grp[grp[args.label] == main_lbl].copy()
            if g.empty: continue
            
            xpos, xtick_labels = prepare_x_for_plot(g[xcol])
            order = np.argsort(xpos)
            xplot = xpos[order]
            mean = g[mean_col].values[order]
                
            ax.plot(xplot, mean, label=f"{exec_name}", color=color, linewidth=2.0)

            if min_col and max_col:
                ymin_vals = g[min_col].values[order]
                ymax_vals = g[max_col].values[order]
                ax.fill_between(xplot, ymin_vals, ymax_vals, color=color, alpha=0.15)

            if xtick_labels is not None:
                last_xtick_labels = xtick_labels

            if args.seq:
                all_x = sorted(g[xcol].dropna().unique())
                if mean_col == "speedup":
                    ax.plot(all_x, all_x, linestyle=":", linewidth=1.5, color=color, alpha=0.6, label=f"{exec_name} ideal")
                elif mean_col == "efficiency":
                    ax.axhline(y=1.0, linestyle=":", linewidth=1.5, color=color, alpha=0.6, label=f"{exec_name} ideal")
                elif mean_col == "real_mean":
                    ideal_y = [ds["t1_global"] / float(x) for x in all_x]
                    ax.plot(all_x, ideal_y, linestyle=":", linewidth=1.5, color=color, alpha=0.6, label=f"{exec_name} ideal")
                elif mean_col == "cpu_mean":
                    ax.axhline(y=ds["t1_global"], linestyle=":", linewidth=1.5, color=color, alpha=0.6, label=f"{exec_name} ideal")

            if args.best and ds["global_best_x"] is not None:
                b_x = float(ds["global_best_x"])
                t_str = "thr" if b_x == 1 else "thrs"
                
                if ax_idx == 0:
                    best_label = f"{exec_name} opt ({ds['global_best_real']:.3g}s @ {b_x:g})"
                else:
                    best_label = f"{exec_name} opt (@ {b_x:g})"
                
                g_best = g[g[xcol] == b_x]
                if not g_best.empty:
                    b_y = g_best[mean_col].values[0]
                    ax.plot(b_x, b_y, marker='x', color=color, markersize=8, markeredgewidth=2,
                            linestyle='None', label=best_label, zorder=10)

        if last_xtick_labels is not None:
            ax.set_xticks(range(len(last_xtick_labels)))
            ax.set_xticklabels(last_xtick_labels, rotation=45)

        ax.set_title(ylabel)
        ax.grid(False)

        if args.xlim:
            ax.set_xlim(args.xlim[0], args.xlim[1])

        if args.log_y:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: "%g" % y))

        if args.log_x:
            ax.set_xscale("log")
            all_numeric_x = []
            for ds in datasets:
                try:
                    all_numeric_x.extend(pd.to_numeric(ds["grp"][xcol], errors="coerce").dropna().unique())
                except:
                    pass
            numeric_x = np.unique(all_numeric_x)
            if numeric_x.size > 0:
                xticks = np.sort(numeric_x.astype(float))
                ax.xaxis.set_major_locator(ticker.FixedLocator(xticks))
                ax.xaxis.set_minor_locator(ticker.NullLocator())
                ax.set_xticks(xticks)
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: "%g" % x))

    for ax in axes[2:]:
        ax.set_xlabel(xcol.replace("_", " ").title())

    n_title_str = " & ".join(sorted(list(n_titles)))
    title_suffix = f" (n={n_title_str})" if n_title_str else ""
    fig.suptitle(f"OpenMP Scaling Analysis Comparison{title_suffix}", fontsize=15, fontweight="bold")
    
    # Step 3: Adaptive Legend Layout
    num_experiments = len(args.experiments)
    # Calculate bottom margin needed: roughly 0.05 per row of legend items
    legend_rows = math.ceil((num_experiments * 3) / 4) # Assuming 4 columns max
    bottom_margin = min(0.35, 0.03 + (legend_rows * 0.04))
    
    fig.tight_layout(rect=[0, 0.07, 1, 0.96])
    
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=min(4, len(labels)))
    
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