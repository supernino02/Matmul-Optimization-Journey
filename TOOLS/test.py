import argparse
import os
import subprocess
import sys
import time
from itertools import product
from pathlib import Path
import pandas as pd  # type: ignore
from config import BUILDS_DIR, RESULTS_DIR


def main():
    # only treat these arguments as "controller" args for the test harness
    p = argparse.ArgumentParser(description="Run builds with parameter sweeps")
    p.add_argument("--builds", nargs="+", required=True,
                   help="Build directories (e.g. gcc_1 icx_1)")
    p.add_argument("--iterations", type=int, default=1,
                   help="Repeat each run K times (default: 1)")
    p.add_argument("--verbose", type=int, choices=[0, 1, 2], default=1,
                   help="Verbosity level: 0=only TIME, 1=also RUNNING, 2=also run outputs (default: 1)")
    p.add_argument("--experiment", type=str, default=None,
                   help="Experiment name; results go to RESULTS/<name>.csv. "
                        "If omitted, no CSV is written and verbosity is set to max.")
    p.add_argument("--build-info", action="store_true",
                   help="Include build metadata columns from builds.csv")

    # parse known args (controller) and capture all other args to delegate
    args, unknown = p.parse_known_args()

    # ── dry-run mode: no experiment → max verbosity, no CSV ──────────────
    write_csv = args.experiment is not None
    if not write_csv:
        args.verbose = 2

    # ── refresh builds.csv ───────────────────────────────────────────────
    build_py = Path(__file__).resolve().parent / "build.py"
    subprocess.run([sys.executable, str(build_py), "--list-builds"],
                   capture_output=True, text=True, check=False)

    # ── load build info if requested ─────────────────────────────────────
    builds_info = {}
    if args.build_info:
        builds_csv = BUILDS_DIR / "builds.csv"
        if builds_csv.exists():
            bdf = pd.read_csv(builds_csv, dtype=str).fillna("")
            for _, brow in bdf.iterrows():
                builds_info[brow["build"]] = brow.drop("build").to_dict()
        else:
            print("[WARN] builds.csv not found; --build-info ignored")

    # ── parse delegated args (unknown) ───────────────────────────────────
    # unknown is a list like ['--n', '128', '256', '-verify', '--max_errors', '16']
    # group tokens into option -> list-of-values (flags get empty list)
    delegated = []  # list of (opt, [values...]) in order
    i = 0
    while i < len(unknown):
        tok = unknown[i]
        if tok.startswith("--"):
            # long option: consume following non-option tokens as values
            key = tok
            vals = []
            j = i + 1
            while j < len(unknown) and not unknown[j].startswith("-"):
                vals.append(unknown[j])
                j += 1
            delegated.append((key, vals))
            i = j
        elif tok.startswith("-"):
            # single-dash token treated as a flag (no following value)
            delegated.append((tok, []))
            i += 1
        else:
            # stray value without an option; treat as positional argument to forward
            delegated.append((None, [tok]))
            i += 1

    # ── validate builds ──────────────────────────────────────────────────
    build_paths = []
    for b in args.builds:
        bd = BUILDS_DIR / b
        exe = bd / "executable"
        if not bd.is_dir():
            print(f"[ERROR] Build directory not found: {bd}")
            return 3
        if not exe.exists():
            print(f"[ERROR] Executable not found in {bd}")
            return 3
        build_paths.append((b, bd, exe))

    # ── validate experiment CSV (only when writing) ───────────────────────
    experiment_csv = None
    header_written = False
    if write_csv:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        experiment_csv = RESULTS_DIR / f"{args.experiment}.csv"
        if experiment_csv.exists():
            print(f"[ERROR] Experiment file already exists: {experiment_csv}")
            return 4

    # ── prepare sweep from delegated args ─────────────────────────────────
    # build column names for delegated args (for CSV)
    col_names = []
    for i, (opt, vals) in enumerate(delegated):
        if opt is None:
            col = f"pos_{i}"
        else:
            col = opt.lstrip('-').replace('-', '_')
        col_names.append(col)

    # identify delegated entries that have multiple values -> sweep over them
    multi_indices = [i for i, (_k, vals) in enumerate(delegated) if len(vals) > 1]
    if multi_indices:
        multi_iters = [delegated[i][1] for i in multi_indices]
        combos = list(product(*multi_iters))
    else:
        combos = [()]


    # ── run ──────────────────────────────────────────────────────────────
    all_output_cols = []  # tracks discovered output columns across all runs for CSV consistency
    
    if any(opt == "--OMP_PROC_BIND" for opt, _ in delegated):
        if not any(opt == "--OMP_PLACES" for opt, _ in delegated):
            os.environ["OMP_PLACES"] = "cores"
            if args.verbose >= 2:
                print("[ENV] Exported OMP_PLACES=cores (default)")
        
    total = len(build_paths) * len(combos) * args.iterations
    print(f"[INFO] Total runs: {total} (builds={len(build_paths)}, variants={len(combos)}, iterations={args.iterations})")
    if not write_csv:
        print(f"[INFO] No --experiment given; running in dry-run mode (verbose=2, no CSV)")
    first_start = time.time()
    done = 0
    for iteration in range(args.iterations): 
        iter_start = time.time()
        for build_name, build_dir, exe_path in build_paths:
            for combo in combos:
                # build delegated args list for this combo and CSV row mapping
                per_args = []
                row = {"executable": build_name}
                env = os.environ.copy()
                
                # fill delegated columns
                for idx, (opt, vals) in enumerate(delegated):
                    col = col_names[idx]
                    if idx in multi_indices:
                        sel = combo[multi_indices.index(idx)]
                        if opt is None:
                            per_args.append(sel)
                            row[col] = sel
                        else:
                            if opt in ("--OMP_PROC_BIND", "--OMP_PLACES"):
                                env_var = opt.lstrip("-")
                                env[env_var] = str(sel)
                                row[col] = sel
                                if args.verbose >= 2:
                                    print(f"[ENV] Exported {env_var}={sel}")
                            else:
                                per_args += [opt, sel]
                                row[col] = sel
                    else:
                        # not swept: include flag or single value(s)
                        if opt is None:
                            # positional
                            row[col] = vals[0] if vals else ""
                            per_args += vals
                        else:
                            if len(vals) == 0:
                                # single-dash flag or explicitly provided flag-like token
                                row[col] = True
                                per_args.append(opt)
                            else:
                                row[col] = vals[0]
                                if opt in ("--OMP_PROC_BIND", "--OMP_PLACES"):
                                    env_var = opt.lstrip("-")
                                    env[env_var] = str(vals[0])
                                    if args.verbose >= 2:
                                        print(f"[ENV] Exported {env_var}={vals[0]}")
                                else:
                                    per_args += [opt] + vals

                cmd = [str(exe_path)] + per_args
                if args.verbose >= 1:
                    print(f"[RUNNING] {build_name} {' '.join(per_args)} ")

                returncode = None
                outmap = {}
                try:
                    res = subprocess.run(
                        cmd, cwd=str(build_dir),
                        capture_output=True, text=True, check=False,
                        env=env
                    )
                    returncode = res.returncode
                except Exception as e:
                    print(f"[ERROR] {exe_path}: {e}")
                    returncode = -1

                if returncode != 0:
                    if returncode == -1:
                        print(f"[ERROR] Failed to launch {exe_path}")
                    else:
                        print(f"[ERROR] rc={returncode}")
                        print(res.stderr)

                # parse k=v output (only when process actually ran)
                if returncode is not None and returncode != -1:
                    for line in res.stdout.splitlines():
                        line = line.strip()
                        if line and "=" in line:
                            k, v = line.split("=", 1)
                            outmap[k.strip()] = v.strip()

                if args.verbose >= 2:
                    print(f"   {outmap}")

                # assemble final row with order: build info, delegated args, outputs
                final_row = {}
                final_row["executable"] = build_name
                final_row["returncode"] = returncode
                if build_name in builds_info:
                    for k, v in builds_info[build_name].items():
                        final_row[k] = v

                # delegated columns in original order
                for col in col_names:
                    final_row[col] = row.get(col, "")

                # outputs — use all_output_cols to keep columns consistent
                for k in all_output_cols:
                    final_row[k] = outmap.get(k, "")
                # discover any new output columns from this run
                for k in outmap:
                    if k not in all_output_cols:
                        all_output_cols.append(k)
                        final_row[k] = outmap[k]

                # write CSV only if experiment was given
                if write_csv:
                    cols = list(final_row.keys())
                    row_df = pd.DataFrame([final_row], columns=cols)
                    row_df.to_csv(experiment_csv, mode="a", index=False,
                                  header=not header_written)
                    header_written = True
                done += 1

            iter_elapsed = time.time() - iter_start
            remaining = args.iterations - (iteration + 1)
            eta = iter_elapsed * remaining
        print(f"[TIME] iter {iteration+1}/{args.iterations}: "
              f"{iter_elapsed:.1f}s elapsed, ETA for remaining {remaining} iteration(s): {eta:.1f}s")

    if write_csv:
        # ── normalize CSV: re-read and rewrite so all rows share the same columns ──
        try:
            df = pd.read_csv(experiment_csv, dtype=str).fillna("")
            df.to_csv(experiment_csv, index=False)
        except Exception as e:
            print(f"[WARN] Could not normalize CSV: {e}")
        print(f"[INFO] Experiment results written to {experiment_csv}, total run-time: {time.time() - first_start:.1f}s")
    else:
        print(f"[INFO] Dry-run complete, total run-time: {time.time() - first_start:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)