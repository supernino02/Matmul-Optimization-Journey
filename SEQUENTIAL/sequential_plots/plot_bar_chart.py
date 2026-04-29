#!/usr/bin/env python3
"""Grouped bar chart from a single experiment CSV.

Each compiler (prefix before the first '_') is a group on the x-axis.
Each compilation type (suffix after the first '_') is a bar within each group,
with consistent colour across all groups.

Usage examples:
  python plots/plot_bar_chart.py --experiment all_comparisons_5000
  python plots/plot_bar_chart.py --experiment all_comparisons_5000 --y cpu_time_seconds
  python plots/plot_bar_chart.py --experiment all_comparisons_5000 \\
      --rename icc "Intel Classic" icx "Intel oneAPI" gcc "GNU" \\
      --rename-type maximum_aligned "Max+Align" 0 "-O0" \\
      --out bar_chart.png
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd                 # type: ignore
import matplotlib.pyplot as plt     # type: ignore

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "experiments"
PLOTS_DIR = ROOT / "plots"


def split_executable(name: str):
    """Split an executable name into (compiler, compilation_type).

    The compiler is the first token before '_', and the compilation type
    is everything after that first '_'.
    Examples:
        icc_maximum_aligned -> ('icc', 'maximum_aligned')
        gcc_3               -> ('gcc', '3')
    """
    parts = name.split("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return name, ""


def apply_renames(text: str, rename_map: dict) -> str:
    """If any key in *rename_map* matches *text* (exact match first, then
    substring / regex), return the replacement value."""
    # exact match takes priority
    if text in rename_map:
        return rename_map[text]
    # pattern / substring match
    for pat, repl in rename_map.items():
        if re.search(pat, text):
            return re.sub(pat, repl, text)
    return text


def build_rename_map(pairs) -> dict:
    """Build a dict from a flat list [k1, v1, k2, v2, …]."""
    if not pairs:
        return {}
    if len(pairs) % 2 != 0:
        print("[ERROR] --rename requires an even number of arguments (pairs of old new)")
        sys.exit(11)
    return {pairs[i]: pairs[i + 1] for i in range(0, len(pairs), 2)}


def main():
    p = argparse.ArgumentParser(
        description="Grouped bar chart — compilers as groups, compilation types as bars"
    )
    p.add_argument("--experiment", default="all_comparisons_5000",
                   help="Experiment name (file in experiments/<name>.csv) (default uses all_comparisons_5000)")
    p.add_argument("--label", default="executable",
                   help="Column whose values are split into compiler + type (default: executable)")
    p.add_argument("--y", default="real_time_seconds",
                   help="Column to use as bar height (default: real_time_seconds)")
    p.add_argument("--rename", nargs="+", metavar="K_V",
                   help="Pairs of pattern replacement applied to the full executable label")
    p.add_argument("--rename-compiler", nargs="+", metavar="K_V",
                   help="Pairs of pattern replacement applied to compiler names")
    p.add_argument("--rename-type", nargs="+", metavar="K_V",
                   help="Pairs of pattern replacement applied to compilation-type names")
    p.add_argument("--out",
                   help="Save plot to this filename (png/pdf); otherwise show interactively")
    p.add_argument("--title", default="Execution Time comparison on n=5000",
                   help="Plot title")
    p.add_argument("--log-y", action="store_true",
                   help="Use logarithmic scale for the y-axis")

    args = p.parse_args()

    # ── rename maps ──────────────────────────────────────────────────────────
    # base defaults requested by user
    DEFAULT_RENAME_FULL = {"blas_like": "ourBLAS -DALIGN"}
    DEFAULT_RENAME_COMP = {"icx": "ICX", "icc": "ICC", "gcc": "GCC"}
    DEFAULT_RENAME_TYPE = {
        "0": "-O0",
        "1": "-O1",
        "2": "-O2",
        "3": "-O3",
        "maximum": "-O3 -xHost",
        "maximum_aligned": "-O3 -xHost -DALIGN",
    }

    rename_full = build_rename_map(args.rename)
    rename_comp = build_rename_map(args.rename_compiler)
    rename_type = build_rename_map(args.rename_type)
    # merge defaults - user-specified values override defaults
    rename_full = {**DEFAULT_RENAME_FULL, **rename_full}
    rename_comp = {**DEFAULT_RENAME_COMP, **rename_comp}
    rename_type = {**DEFAULT_RENAME_TYPE, **rename_type}

    # ── load CSV ─────────────────────────────────────────────────────────────
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

    ycol = args.y
    label_col = args.label

    for col in (label_col, ycol):
        if col not in df.columns:
            print(f"[ERROR] Column '{col}' not in CSV. Available: {', '.join(df.columns)}")
            sys.exit(5)

    df[ycol] = pd.to_numeric(df[ycol], errors="coerce")
    df = df[df[ycol].notna()].copy()

    # ── split labels into compiler / type ────────────────────────────────────
    # Apply full-label renames first (before splitting)
    df["_label"] = df[label_col].astype(str).apply(lambda s: apply_renames(s, rename_full))
    df[["_compiler", "_type"]] = df["_label"].apply(
        lambda s: pd.Series(split_executable(s))
    )

    # Apply per-part renames
    df["_compiler"] = df["_compiler"].apply(lambda s: apply_renames(s, rename_comp))
    df["_type"] = df["_type"].apply(lambda s: apply_renames(s, rename_type))

    # ── aggregate (mean) in case there are repeated runs ─────────────────────
    agg = df.groupby(["_compiler", "_type"])[ycol].mean().reset_index()

    compilers = list(dict.fromkeys(df["_compiler"]))       # preserve order
    comp_types = list(dict.fromkeys(df["_type"]))           # preserve order

    # ── sort comp_types by overall mean height (ascending) ───────────────────
    type_means = agg.groupby("_type")[ycol].mean()
    comp_types.sort(key=lambda t: type_means.get(t, 0), reverse=True)

    # ── plot ─────────────────────────────────────────────────────────────────
    n_groups = len(compilers)
    n_bars = len(comp_types)
    x = np.arange(n_groups)
    total_width = 0.8
    bar_width = total_width / max(1, n_bars)

    fig, ax = plt.subplots(figsize=(max(8, n_groups * 2.2), 6))

    for i, ctype in enumerate(comp_types):
        heights = []
        for comp in compilers:
            row = agg[(agg["_compiler"] == comp) & (agg["_type"] == ctype)]
            heights.append(row[ycol].values[0] if not row.empty else 0)
        offsets = x - (total_width - bar_width) / 2 + i * bar_width
        bars = ax.bar(offsets, heights, width=bar_width, label=ctype)

    ax.set_xticks(x)
    ax.set_xticklabels(compilers, rotation=0, ha="center", fontsize=11)
    ax.set_ylabel("Execution Time")
    ax.set_xlabel("Compiler")
    ax.set_title(args.title)
    # legend at top-right corner, arranged in two columns (wraps into rows)
    # compute number of columns to try keep maximum ~4 rows
    n_items = len(comp_types)
    ncol = 2
    # optional: adjust ncol if very few items
    if n_items <= 4:
        ncol = 1
    ax.legend(title="Executable", loc="upper right",
              ncol=ncol, frameon=False, columnspacing=1.0, labelspacing=0.5)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    if args.log_y:
        ax.set_yscale("log")

    plt.tight_layout()

    # ── output ───────────────────────────────────────────────────────────────
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
