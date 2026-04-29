#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot OpenMP schedule and block size experiments as an aggressive log-Y bar chart.

Usage examples:
  python plot_blocks_aggressive.py --experiment blas_like_scheduling_blocks
  python plot_blocks_aggressive.py --experiment blas_like_scheduling_blocks --out aggressive_blocks.png

This script aggregates the real execution time by chunk size and schedule, and applies
an aggressive base-2 logarithmic scale to the Y-axis (clamped at 14s) to expose micro-differences 
at the lower bounds while retaining massive outliers.

Mandatory flags:
  --experiment EXPERIMENT  Experiment name (file in experiments/<name>.csv)

Optional flags:
  --out FILENAME           Save plot to this filename (png/pdf); otherwise show interactively
"""
import argparse
import sys
from pathlib import Path

import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import matplotlib.ticker as ticker  # type: ignore

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "experiments"
PLOTS_DIR = ROOT / "plots"


def main():
    p = argparse.ArgumentParser(description="Plot OpenMP chunk sizes (Aggressive Log Bar Chart)")
    p.add_argument("--experiment", required=True,
                   help="Experiment name (file in experiments/<name>.csv)")
    p.add_argument("--out",
                   help="Save plot to this filename (png/pdf); otherwise show interactively")

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

    # Validate required columns
    xcol = "block_size"
    for col in ["schedule", xcol, "real_time_seconds"]:
        if col not in df.columns:
            print(f"[ERROR] Column '{col}' not found. Available: {', '.join(df.columns)}")
            sys.exit(5)

    # -- Extract dynamic 'n' and 'threads' value for title -------------------
    title_suffix = ""
    if "n" in df.columns and "threads" in df.columns:
        n_vals = df["n"].dropna().unique()
        t_vals = df["threads"].dropna().unique()
        if len(n_vals) == 1 and len(t_vals) == 1:
            title_suffix = f" (n={n_vals[0]}, threads={t_vals[0]})"

    # -- Data Preparation & Aggregation --------------------------------------
    # Map the OpenMP numeric schedules to their actual names
    sched_map = {1: 'static', 2: 'dynamic', 3: 'guided'}
    df['schedule_name'] = df['schedule'].map(sched_map).fillna(df['schedule'])

    # Aggregate by computing the mean execution time
    agg_df = df.groupby([xcol, 'schedule_name'])['real_time_seconds'].mean().unstack()
    
    # Ensure consistent column ordering for the plot
    columns_present = [col for col in ['static', 'dynamic', 'guided'] if col in agg_df.columns]
    agg_df = agg_df[columns_present]

    # -- Plot Configuration --------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))

    agg_df.plot(
        kind='bar', 
        ax=ax, 
        width=0.85, 
        color=['#A6CEE3', '#1F78B4', '#B2DF8A']
    )

    ax.set_title(f"Average Real Time by Block Size and Schedule{title_suffix}\n", 
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Real Time (s)", fontsize=12)
    ax.set_xlabel("Chunk Size", fontsize=12)

    # --- AGGRESSIVE LOG SCALE ---
    # Clamp bottom explicitly at 14 to stretch the 15-18 second range
    ax.set_ylim(14, 140)
    ax.set_yscale('log', basey=2)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))

    plt.xticks(rotation=0)
    
    # Grid lines to guide the eye through the logarithmic jumps
    ax.grid(axis='y', which='major', linestyle='-', alpha=0.5)
    ax.grid(axis='y', which='minor', linestyle='--', alpha=0.2)


    # Add explicit value labels on top of the bars to guarantee readability
    # Find the global minimum execution time among all drawn bars
    valid_heights = [p.get_height() for p in ax.patches if pd.notnull(p.get_height()) and p.get_height() > 0]
    min_height = min(valid_heights) if valid_heights else None

    # Only add the label on top of the bar representing the minimum time
    for p in ax.patches:
        height = p.get_height()
        if height == min_height:
            # Added a bold font and red color to make the optimal value pop
            ax.annotate(f'{height:.2f} seconds',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom',
                        xytext=(0, 4), # 4 points vertical offset
                        textcoords='offset points',
                        fontsize=10, fontweight='bold', color='red', rotation=90)

    # -- Layout and Legend Formatting (Inspired by plot_openmp.py) -----------
    # Remove the default axis legend created by pandas
    if ax.get_legend() is not None:
        ax.get_legend().remove()

    # Restrict the subplot to leave room at the bottom for the horizontal legend
    fig.tight_layout(rect=[0, 0.08, 1, 0.96])
    
    # Extract handles and generate the unified Figure legend at the bottom center
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, title="OpenMP Schedule", loc='lower center', 
               bbox_to_anchor=(0.5, 0.0), ncol=len(labels))

    # -- Output Handling -----------------------------------------------------
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
            
        plt.savefig(out_file, dpi=300)
        print(f"[OK] Plot saved to {out_file}")
    else:
        plt.show()

if __name__ == "__main__":
    main()