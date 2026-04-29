#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot OpenMP Binding and Placement experiments as an aggressive log-Y bar chart.

Usage examples:
  python plot_binding.py --experiment binding_results --group places
  python plot_binding.py --experiment binding_results --group bind --out my_plot.png
"""
import argparse
import sys
from pathlib import Path

import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import matplotlib.ticker as ticker  # type: ignore

# Assume standard project structure
ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "experiments"
PLOTS_DIR = ROOT / "plots"

def main():
    p = argparse.ArgumentParser(description="Plot OpenMP Binding Policies")
    p.add_argument("--experiment", required=True,
                   help="Experiment name (file in experiments/<name>.csv)")
    p.add_argument("--group", choices=["places", "bind"], default="places",
                   help="Select which variable serves as the primary X-axis (default: places).")
    p.add_argument("--out",
                   help="Save plot to this filename (png/pdf); otherwise show interactively")

    args = p.parse_args()

    # Load Data
    csv_path = RESULTS_DIR / f"{args.experiment}.csv"
    if not csv_path.exists():
        print(f"[ERROR] Experiment file not found: {csv_path}")
        # For testing, fallback to local path if not in the experiments dir
        csv_path = Path(f"{args.experiment}.csv")
        if not csv_path.exists():
            sys.exit(2)

    try:
        df = pd.read_csv(csv_path, comment="/")
    except Exception as e:
        print(f"[ERROR] Could not read CSV: {e}")
        sys.exit(3)

    for col in ["OMP_PLACES", "OMP_PROC_BIND", "real_time_seconds"]:
        if col not in df.columns:
            print(f"[ERROR] Required column '{col}' not found.")
            sys.exit(5)

    # Clean data: drop NAs in target columns
    df = df.dropna(subset=["OMP_PLACES", "OMP_PROC_BIND", "real_time_seconds"])

    # -- Extract dynamic contextual parameters for the title --
    # Exclude metrics, sweep variables, and boilerplate columns
    ignore_cols = {"OMP_PLACES", "OMP_PROC_BIND", "real_time_seconds", "cpu_time_seconds", "executable", "returncode"}
    
    constant_params = []
    for col in df.columns:
        if col not in ignore_cols:
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) == 1:
                val = unique_vals[0]
                
                # Optional: Make OpenMP schedules readable instead of numeric
                if col == "schedule":
                    sched_map = {1: 'static', 2: 'dynamic', 3: 'guided', 4: 'auto'}
                    try:
                        val = sched_map.get(int(val), val)
                    except ValueError:
                        pass # Leave as is if it's already a string

                constant_params.append(f"{col}={val}")

    title_suffix = f"\n({', '.join(constant_params)})" if constant_params else ""

    # Define logical ordering
    places_order = ['threads', 'cores', 'sockets']
    bind_order = ['master', 'close', 'spread']

    # Apply configuration based on the --group flag
    if args.group == "places":
        group_cols = ['OMP_PLACES', 'OMP_PROC_BIND']
        idx_order = [p for p in places_order if p in df['OMP_PLACES'].values]
        col_order = [b for b in bind_order if b in df['OMP_PROC_BIND'].values]
        x_label = "OMP_PLACES (Granularity of Hardware Locations)"
        legend_title = "OMP_PROC_BIND"
        # Colors for master (red), close (blue), spread (green)
        colors = ['#fb9a99', '#a6cee3', '#b2df8a']
    else:
        group_cols = ['OMP_PROC_BIND', 'OMP_PLACES']
        idx_order = [b for b in bind_order if b in df['OMP_PROC_BIND'].values]
        col_order = [p for p in places_order if p in df['OMP_PLACES'].values]
        x_label = "OMP_PROC_BIND (Thread Distribution Policy)"
        legend_title = "OMP_PLACES"
        # Colors for threads (blue), cores (green), sockets (red)
        colors = ['#a6cee3', '#b2df8a', '#fb9a99']

    # Aggregate by taking the mean across iterations
    agg_df = df.groupby(group_cols)['real_time_seconds'].mean().unstack()
    agg_df = agg_df.reindex(index=idx_order, columns=col_order)

    # Plot Configuration
    fig, ax = plt.subplots(figsize=(10, 6))
    
    agg_df.plot(kind='bar', ax=ax, width=0.8, color=colors[:len(col_order)])

    ax.set_title(f"Average Real Time by Thread Placement and Binding Policy{title_suffix}\n", 
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Real Time (s)", fontsize=12)
    ax.set_xlabel(x_label, fontsize=12)

    # --- AGGRESSIVE LOG SCALE ---
    ax.set_ylim(14, 150)
    # Using basey=2 for compatibility with Matplotlib < 3.3
    ax.set_yscale('log', base=2)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))

    plt.xticks(rotation=0)
    ax.grid(axis='y', which='major', linestyle='-', alpha=0.5)
    ax.grid(axis='y', which='minor', linestyle='--', alpha=0.2)

    # Manual annotation for explicit value labels
    for p_patch in ax.patches:
        height = p_patch.get_height()
        if pd.notnull(height) and height > 0:
            ax.annotate(f'{height:.1f}',
                        (p_patch.get_x() + p_patch.get_width() / 2., height),
                        ha='center', va='bottom',
                        xytext=(0, 4), # 4 points vertical offset
                        textcoords='offset points',
                        fontsize=9)

    # Extract handles and generate the unified Figure legend at the bottom center
    if ax.get_legend() is not None:
        ax.get_legend().remove()

    fig.tight_layout(rect=[0, 0.08, 1, 0.96])
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, title=legend_title, loc='lower center', 
               bbox_to_anchor=(0.5, 0.0), ncol=len(labels))

    # Output Handling
    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_file = PLOTS_DIR / out_path
        else:
            out_file = out_path
            
        out_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_file, dpi=150)
        print(f"[OK] Plot saved to {out_file}")
    else:
        plt.show()

if __name__ == "__main__":
    main()