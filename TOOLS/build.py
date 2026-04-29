"""Compile & manage builds for the HPC matrix-multiply project."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd  # type: ignore

from config import (
    BUILDS_DIR,
    COMMON_SRCS,
    CUDA_DIR,
    CUDA_KERNELS_DIR,
    DEFAULT_CFLAGS,
    GCC_EXTRA_FLAGS,
    ICC_EXTRA_FLAGS,
    ICX_EXTRA_FLAGS,
    SRC_DIR,
    VERSIONS_DIR,
    SETVARS,
)

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


# ─── helpers ────────────────────────────────────────────────────────────────

def load_builds():
    """Return list of (build_name, metadata_dict) for every existing build."""
    builds = []
    if not BUILDS_DIR.is_dir():
        return builds
    for d in sorted(BUILDS_DIR.iterdir()):
        meta = d / "metadata.json"
        if d.is_dir() and meta.exists():
            with open(meta) as f:
                builds.append((d.name, json.load(f)))
    return builds


def save_metadata(build_dir: Path, version: str, compiler: str, flags: list):
    """Write metadata.json into build_dir."""
    meta = {
        "compiler": compiler,
        "flags_list": flags,
        "target": "executable",
        "version": version,
    }
    with open(build_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")


def create_numbered_build_dir(compiler: str) -> Path:
    """Create and return a new build dir like builds/gcc_4."""
    BUILDS_DIR.mkdir(parents=True, exist_ok=True)
    idx = 1
    while True:
        d = BUILDS_DIR / f"{compiler}_{idx}"
        if not d.exists():
            d.mkdir()
            return d
        idx += 1


def create_build_dir(name: str, compiler: str) -> Path:
    """Create and return a build dir.

    If `name` is truthy, create `builds/<name>` (error if exists).
    Otherwise fall back to numbered name like `compiler_N`.
    """
    BUILDS_DIR.mkdir(parents=True, exist_ok=True)
    if name:
        d = BUILDS_DIR / name
        if d.exists():
            return None # type: ignore
        d.mkdir()
        return d
    return create_numbered_build_dir(compiler)


def find_matching_build(version: str, compiler: str, flags: list):
    """Return build name if an identical build already exists, else None."""
    for name, meta in load_builds():
        if (
            meta.get("version") == version
            and meta.get("compiler") == compiler
            and sorted(meta.get("flags_list", [])) == sorted(flags)
        ):
            return name
    return None


# ─── compile ────────────────────────────────────────────────────────────────

def compile_version(version: str, compiler: str, extra_params: str, report: bool, native: bool = False, unroll: bool = False, name: str = None, best_flags: bool = False): # type: ignore
    """Compile a version and store the result in a new build directory."""
    src_file = VERSIONS_DIR / f"{version}.c"
    if not src_file.exists():
        print(f"[ERROR] Source file not found: {src_file}")
        return 1

    # assemble flags
    flags = list(DEFAULT_CFLAGS)
    link_libs = []  # libraries that must appear after sources in the link command
    if compiler == "icc":
        flags += ICC_EXTRA_FLAGS
    elif compiler == "icx":
        flags += ICX_EXTRA_FLAGS
    else:
        flags += GCC_EXTRA_FLAGS
    if extra_params:
        flags += extra_params.split()

    if best_flags:
        native = True
        unroll = True
        flags.append("-DALIGN")

    # --native: compiler-specific native/host tuning
    if native:
        flags += ["-O3", "-ipo"]  # default to -march=native for all compilers
        if compiler in ("icx", "icc"):
            flags.append("-xHost")
        else:
            flags.append("-march=native")

    # --unroll: enable aggressive inlining/unrolling (compiler-specific mapping)
    if unroll:
        if compiler == "icc":
            flags += ["-inline-level=2", "-unroll-aggressive", "-unroll=4"]
        elif compiler == "icx":
            # ICX uses LLVM-style flags
            flags += ["-mllvm", "-unroll-count=4", "-funroll-loops"]
        else:
            # GCC: use equivalent options for inlining and loop unrolling
            flags += ["-finline-functions", "-funroll-loops", "-funroll-all-loops"]

    # Automatically add oneMKL link flags if the version is "mkl"
    if version == "mkl":
        # Require -DSEQUENTIAL or -DPARALLEL in --params
        params = (extra_params or "").split()
        has_seq = "-DSEQUENTIAL" in params
        has_par = "-DPARALLEL" in params
        if not has_seq and not has_par:
            exit("[ERROR] MKL version requires threading mode. Please specify -DSEQUENTIAL or -DPARALLEL via --params")

        mkl_lib = "/opt/intel/oneapi/mkl/latest/lib/intel64"
        mkl_inc = "/opt/intel/oneapi/mkl/latest/include"
        flags.append(f"-I{mkl_inc}")

        # Static link: interface + threading-layer + core (wrapped in --start/end-group
        # because the three archives have circular dependencies)
        if has_seq:
            thread_lib = f"{mkl_lib}/libmkl_sequential.a"
        else:
            thread_lib = f"{mkl_lib}/libmkl_intel_thread.a"

        link_libs = [
            "-Wl,--start-group",
            f"{mkl_lib}/libmkl_intel_lp64.a",
            thread_lib,
            f"{mkl_lib}/libmkl_core.a",
            "-Wl,--end-group",
            "-lpthread", "-lm", "-ldl",
        ]
        # Parallel MKL needs the Intel OpenMP runtime
        if has_par:
            iomp_dir = "/opt/intel/oneapi/compiler/2023.2.1/linux/compiler/lib/intel64_lin"
            link_libs += [f"-L{iomp_dir}", "-liomp5"]

    elif version == "openblas":
        # Require -DSEQUENTIAL or -DPARALLEL in --params
        has_seq = "-DSEQUENTIAL" in (extra_params or "").split()
        has_par = "-DPARALLEL" in (extra_params or "").split()
        if not has_seq and not has_par:
            exit(f"[ERROR] OpenBLAS version requires threading mode. Please specify -DSEQUENTIAL or -DPARALLEL via --params")

        # Get the absolute path to the current working directory
        base_dir = os.getcwd()
        openblas_dir = f"{base_dir}/local_openblas"

        flags.append(f"-I{openblas_dir}/include")     # Where to find cblas.h
        # Libraries must come AFTER source files in the link command (linker is left-to-right)
        link_libs = [
            f"{openblas_dir}/lib/libopenblas.a",      # Statically link OpenBLAS
            "-lm", "-lpthread", "-lgfortran",         # Dependencies required for static OpenBLAS
        ]

    # duplicate check
    dup = find_matching_build(version, compiler, flags)
    if dup:
        print(f"[SKIP] Identical build already exists: {dup}")
        return 0

    # if a custom name was provided, ensure it's not already taken
    if name:
        name_dir = BUILDS_DIR / name
        if name_dir.exists():
            print(f"[ERROR] Build name already exists: {name}")
            return 1

    build_dir = create_build_dir(name, compiler)
    if build_dir is None:
        print(f"[ERROR] Could not create build directory: {name}")
        return 1
    exe_path = build_dir / "executable"

    srcs = [str(src_file)] + [str(s) for s in COMMON_SRCS]
    cmd = [compiler] + flags + srcs + ["-o", str(exe_path), f"-I{SRC_DIR}"] + link_libs

    if report:
        if compiler in ("icc"):
            cmd += ["-qopt-report=5", "-qopt-report-phase=vec,loop",f"-qopt-report-file={exe_path}.opt.txt"]
        elif compiler in ("icx"):
            cmd += ["-qopt-report=3",
                    f"-qopt-report-file={exe_path}.opt.txt",
                    "-Rpass=loop-unroll|loop-interchange|loop-vectorize|loop-tile",
                    "-Rpass-missed=loop-unroll|loop-interchange|loop-vectorize|loop-tile",
                    "-Rpass-analysis=loop-unroll|loop-interchange|loop-vectorize|loop-tile"]
        else:
            cmd += [f"-fopt-info-all={exe_path}.opt.txt"]

    print(f"[INFO] Compiling {version} with {compiler}")
    print(f"       flags: {' '.join(flags)}")
    print(f"       dir:   {build_dir.name}")

    try:
        # run the compiler with cwd=build_dir so icx/icc create their report files
        # (yaml, opt reports, etc.) inside the build directory
        res = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=str(build_dir))
    except FileNotFoundError:
        print(f"[ERROR] Compiler not found: {compiler}")
        build_dir.rmdir()
        return 1

    if res.returncode != 0:
        print(f"[ERROR] Compilation failed (rc={res.returncode})")
        print(res.stderr)
        # clean up empty dir on failure
        if not any(build_dir.iterdir()):
            build_dir.rmdir()
        return 1

    if res.stderr:
        print(res.stderr)

    save_metadata(build_dir, version, compiler, flags)
    print(f"[OK]   {build_dir.name}/executable")
    return 0


# ─── CUDA compile ───────────────────────────────────────────────────────────

def compile_cuda(version: str, extra_params: str, name: str = None):  # type: ignore
    """Compile a CUDA kernel version.

    Steps:
      1. nvcc -c  kernel .cu            → kernel.o        (in build dir)
      2. nvcc -c  cuda_wrapper.cu        → cuda_wrapper.o  (in build dir)
      3. gcc  -c  common C sources       → *.o             (in build dir)
      4. nvcc     link everything         → executable
      5. remove intermediate .o files
    """
    kernel_file = CUDA_KERNELS_DIR / f"{version}.cu"
    if not kernel_file.exists():
        print(f"[ERROR] CUDA kernel not found: {kernel_file}")
        return 1

    wrapper_file = CUDA_DIR / "cuda_wrapper.cu"
    if not wrapper_file.exists():
        print(f"[ERROR] cuda_wrapper.cu not found: {wrapper_file}")
        return 1

    # assemble flags list (for metadata / duplicate check)
    nvcc_extra = ["-arch=sm_75"]
    gcc_flags = ["-fopenmp", "-Wall", "-Wextra", "-DCUDA"]
    if extra_params:
        for flag in extra_params.split():
            gcc_flags.append(flag)
            nvcc_extra.append(flag)
    all_flags = gcc_flags + nvcc_extra  # stored in metadata for duplicate detection

    # duplicate check (same version + compiler + flags)
    dup = find_matching_build(version, "nvcc", all_flags)
    if dup:
        print(f"[SKIP] Identical build already exists: {dup}")
        return 0

    # if a custom name was provided, ensure it's not already taken
    if name:
        name_dir = BUILDS_DIR / name
        if name_dir.exists():
            print(f"[ERROR] Build name already exists: {name}")
            return 1

    build_dir = create_build_dir(name, "nvcc")
    if build_dir is None:
        print(f"[ERROR] Could not create build directory: {name}")
        return 1
    exe_path = build_dir / "executable"

    print(f"[INFO] Compiling CUDA kernel '{version}'")
    print(f"       gcc flags:  {' '.join(gcc_flags)}")
    print(f"       nvcc extra: {' '.join(nvcc_extra) if nvcc_extra else '(none)'}")
    print(f"       dir:        {build_dir.name}")

    inc = f"-I{SRC_DIR}"

    # ── step 1: compile kernel .cu ──────────────────────────────────────
    kernel_obj = build_dir / f"{version}.o"
    cmd1 = ["nvcc", "-c", str(kernel_file), "-o", str(kernel_obj), inc] + nvcc_extra

    # ── step 2: compile cuda_wrapper.cu ─────────────────────────────────
    wrapper_obj = build_dir / "cuda_wrapper.o"
    cmd2 = ["nvcc", "-c", str(wrapper_file), "-o", str(wrapper_obj), inc] + nvcc_extra

    # ── step 3: compile common C sources with gcc ───────────────────────
    c_objs = []
    c_cmds = []
    for src in COMMON_SRCS:
        obj = build_dir / (src.stem + ".o")
        c_objs.append(obj)
        c_cmds.append(["gcc"] + gcc_flags + ["-c", str(src), "-o", str(obj), inc])

    # ── step 4: link with nvcc ──────────────────────────────────────────
    all_objs_paths = c_objs + [kernel_obj, wrapper_obj]
    cmd_link = ["nvcc", "-Xcompiler", "-fopenmp"] + [str(o) for o in all_objs_paths] + ["-o", str(exe_path)]
    
    # Add cuBLAS library if using cu_blas kernel
    if version == "cu_blas":
        cmd_link.append("-lcublas")

    # run all steps sequentially
    steps = [("kernel", cmd1), ("wrapper", cmd2)] + [(src.name, cmd) for src, cmd in zip(COMMON_SRCS, c_cmds)] + [("link", cmd_link)]

    for step_name, cmd in steps:
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=str(build_dir))
        except FileNotFoundError as e:
            print(f"[ERROR] Command not found during '{step_name}' step: {e}")
            return 1
        if res.returncode != 0:
            print(f"[ERROR] '{step_name}' step failed (rc={res.returncode})")
            print(res.stderr)
            return 1
        if res.stderr:
            print(res.stderr, end="")

    # ── step 5: clean up intermediate .o files ──────────────────────────
    for obj in all_objs_paths:
        if obj.exists():
            obj.unlink()

    save_metadata(build_dir, version, "nvcc", all_flags)
    print(f"[OK]   {build_dir.name}/executable")
    return 0


# ─── CSV summary ────────────────────────────────────────────────────────────

def write_builds_csv():
    """(Re)generate builds/builds.csv from metadata.json files and print it."""
    builds = load_builds()
    if not builds:
        print("[INFO] No builds found.")
        return

    rows = []
    all_flag_keys = set()

    for name, meta in builds:
        row = {
            "build": name,
            "version": meta.get("version", ""),
            "compiler": meta.get("compiler", ""),
        }
        for f in meta.get("flags_list", []):
            if f.startswith("-"):
                key = f.split("=")[0] if "=" in f else f
                val = f.split("=", 1)[1] if "=" in f else True

                # flags like -std=c11 → key=-std, val=c11
                # flags like -O3     → key=-O3, val=True
                row[key] = val
                all_flag_keys.add(key)
        rows.append(row)

    df = pd.DataFrame(rows)
    # consistent column order: build, version, compiler, then sorted flags
    cols = ["build", "version", "compiler"] + sorted(all_flag_keys)
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    df = df[cols].fillna("")

    csv_path = BUILDS_DIR / "builds.csv"
    df.to_csv(csv_path, index=False)
    print(df.to_csv(index=False), end="")


# ─── main ───────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Compile & manage builds")
    p.add_argument("--compiler",               help="Compiler executable (gcc, icx, icc, nvcc)")
    p.add_argument("--version", "-V",          help="Version name (e.g. original, mkl, openblas, naive for CUDA)")
    p.add_argument("--name", "-n", help="Custom build directory name (e.g. my-build)")
    p.add_argument("--best-flags", action="store_true", help="Shortcut for maximum optimization: -O3 -ipo -unroll (icx/icc) or -O3 -ipo -funroll-loops (gcc)")
    p.add_argument("--params", "-p", default="", help="Extra compiler flags (quoted string). For MKL/OpenBLAS: -DSEQUENTIAL or -DPARALLEL is required.")
    p.add_argument("--report", action="store_true", help="Generate vectorisation report")
    p.add_argument("--native", action="store_true", help="Enable native/host architecture tuning (-march=native for gcc, -xhost for icx/icc)")
    p.add_argument("--unroll", action="store_true", help="Enable aggressive inlining/unrolling (compiler-specific mapping)")
    p.add_argument("--list-builds", action="store_true", help="Rebuild builds.csv and print it")
    args = p.parse_args()

    import_oneapi_setvars()

    rc = 0
    if args.version and args.compiler:
        if args.compiler == "nvcc":
            if args.report or args.native or args.unroll:
                print("[WARN] --report, --native, and --unroll are ignored for CUDA builds.")
            rc = compile_cuda(args.version, args.params, args.name)
        else:
            rc = compile_version(args.version, args.compiler, args.params, args.report, args.native, args.unroll, args.name, args.best_flags)
        load_builds()

    elif args.version or args.compiler:
        print("[ERROR] Both --version and --compiler are required to compile.")
        rc = 2

    if args.list_builds:
        write_builds_csv()

    if not args.version and not args.compiler and not args.list_builds:
        p.print_help()

    return rc


if __name__ == "__main__":
    sys.exit(main() or 0)
