"""Run profiling tools (e.g., Intel Advisor) on built executables."""
import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
import os

from config import BUILDS_DIR, RESULTS_DIR,SETVARS


def run_intel_advisor(exe_path: Path, build_dir: Path, experiment: str, extra_args: list):
    """
    Run Intel Advisor collections (survey, tripcounts, roofline) and generate a report.
    Creates an HTML report in RESULTS_DIR and saves advisor collect logs in the project_dir
    before cleaning it up.
    """
    def import_oneapi_setvars():
        """Source Intel setvars.sh and merge its env into the current process."""
        if not os.path.isfile(SETVARS):
            return
        try:
            out = subprocess.run(
                ["bash", "-c", f"source {SETVARS} --force > /dev/null 2>&1 && env"],
                capture_output=True, text=True, check=True,
            )
        except Exception:
            return
        for line in out.stdout.splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                os.environ[k] = v

    import_oneapi_setvars()

    # Create a temporary project directory for Advisor
    project_dir = Path(tempfile.mkdtemp(prefix="advisor_"))

    # prepare executable invocation
    exe_cmd = [str(exe_path)] + extra_args

    # helper to run advisor collects and write logs
    def _run_collect(args_list, logfile_name):
        log_path = project_dir / logfile_name
        print(f"[RUNNING] {' '.join(args_list)}")
        res = subprocess.run(
            args_list,
            cwd=str(build_dir),
            capture_output=True,
            text=True,
            check=False,
        )
        # write stdout/stderr to a log in the project dir for inspection
        try:
            with open(log_path, "w") as lf:
                lf.write("--- STDOUT ---\n")
                lf.write(res.stdout or "")
                lf.write("\n--- STDERR ---\n")
                lf.write(res.stderr or "")
        except Exception:
            pass
        return res.returncode, res

    try:
        # 1) roofline macro (esegue automaticamente survey + tripcounts + cache sim per CPU)
        roof_cmd = [
            "advisor",
            "--collect=roofline",
            "--no-profile-gpu",           # <-- FORZA IL PROFILING SULLA CPU
            "--enable-cache-simulation",
            f"--project-dir={project_dir}",
            "--",
        ] + exe_cmd
        rc, _ = _run_collect(roof_cmd, "collect_roofline.log")
        if rc != 0:
            print(f"[ERROR] Advisor CPU roofline collection failed (rc={rc})")
            return 1

        # 2) Generazione del report HTML interattivo
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        report_path = RESULTS_DIR / f"{experiment}.html"

        report_cmd = [
            "advisor",
            "--report=roofline",             # <-- RICHIEDE ESPLICITAMENTE IL ROOFLINE HTML
            f"--project-dir={project_dir}",
            f"--report-output={report_path}",
        ]

        print(f"[RUNNING] Generating HTML report...")
        rc, res = _run_collect(report_cmd, "report_generation.log")
        if rc != 0:
            print(f"[ERROR] Advisor report generation failed (rc={rc})")
            # Stampa stderr per feedback immediato
            if res and res.stderr:
                print(res.stderr)
            return 2

        print(f"[INFO] Report written to {report_path}")
        return 0

    finally:
        # Clean up project directory
        if project_dir.exists():
            print(f"[CLEANUP] Removing project directory: {project_dir}")
            shutil.rmtree(project_dir, ignore_errors=True)


def run_valgrind(exe_path: Path, build_dir: Path, experiment: str, extra_args: list):
    """
    Run Valgrind cachegrind on the executable and save output to RESULTS_DIR/{experiment}.valgrind.txt
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_DIR / f"{experiment}.valgrind.txt"

    # build command: valgrind --tool=cachegrind --log-file=<path> -- <exe> [args...]
    cmd = ["valgrind", "--tool=cachegrind", f"--log-file={report_path}", "--", str(exe_path)] + extra_args
    print(f"[RUNNING] Valgrind cachegrind: {' '.join(cmd)}")
    try:
        res = subprocess.run(cmd, cwd=str(build_dir), capture_output=True, text=True, check=False)
    except FileNotFoundError:
        print("[ERROR] valgrind not found on PATH")
        return 3

    if res.returncode != 0:
        print(f"[ERROR] valgrind returned rc={res.returncode}")
        print(res.stderr)
        return 1

    print(f"[INFO] Valgrind output written to {report_path}")
    return 0


def main():
    p = argparse.ArgumentParser(
        description="Run profiling tools on built executables"
    )
    # tool flags: allow selecting multiple
    p.add_argument(
        "--intel-advisor",
        action="store_true",
        help="Run Intel Advisor (roofline, tripcounts, survey)"
    )
    p.add_argument(
        "--valgrind",
        action="store_true",
        help="Run Valgrind cachegrind"
    )
    p.add_argument(
        "--build",
        required=True,
        help="Build directory name (e.g. icx_base)",
    )
    p.add_argument(
        "--experiment",
        required=True,
        help="Experiment name; report goes to RESULTS/<name>",
    )

    # Parse known args and capture remaining args to pass to executable
    args, extra_args = p.parse_known_args()

    # ── validate build ───────────────────────────────────────────────────
    build_dir = BUILDS_DIR / args.build
    exe_path = build_dir / "executable"

    if not build_dir.is_dir():
        print(f"[ERROR] Build directory not found: {build_dir}")
        return 3

    if not exe_path.exists():
        print(f"[ERROR] Executable not found in {build_dir}")
        return 3

    # ensure at least one tool selected
    if not (args.intel_advisor or args.valgrind):
        print("[ERROR] No tool selected. Use --intel-advisor and/or --valgrind")
        return 2

    # ── validate experiment output ───────────────────────────────────────
    # check for any outputs that would be overwritten
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    conflict_paths = []
    if args.intel_advisor:
        conflict_paths.append(RESULTS_DIR / f"{args.experiment}.html")
    if args.valgrind:
        conflict_paths.append(RESULTS_DIR / f"{args.experiment}.valgrind.txt")

    for pth in conflict_paths:
        if pth.exists():
            print(f"[ERROR] Experiment output already exists: {pth}")
            return 4

    # ── run the selected tools (allow multiple) ──────────────────────────
    rc_total = 0
    if args.intel_advisor:
        rc = run_intel_advisor(exe_path, build_dir, args.experiment, extra_args)
        if rc != 0:
            rc_total = rc_total or rc
    if args.valgrind:
        rc = run_valgrind(exe_path, build_dir, args.experiment, extra_args)
        if rc != 0:
            rc_total = rc_total or rc

    return rc_total


if __name__ == "__main__":
    sys.exit(main() or 0)
