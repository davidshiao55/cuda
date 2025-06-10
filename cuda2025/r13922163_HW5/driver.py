#!/usr/bin/env python3

import argparse
import subprocess
import shlex
import filecmp
import sys
from pathlib import Path
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt

BIN_CPU    = "./hw5_cpu"
BIN_SINGLE = "./hw5_single"
BIN_MULTI  = "./hw5_multi"

CPU_OUT    = "output_cpu.csv"
SGL_OUT    = "output_single.csv"
MULT_OUT   = "output_multi.csv"

BLOCK_LIST = [4, 8, 16, 32]          # automatically tested block sizes


# ----------------------------------------------------------------------
def run(cmd):
    """Run a shell command, echo its output, return the full stdout."""
    proc = subprocess.Popen(
        shlex.split(cmd),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1)
    out_lines = []
    for line in proc.stdout:
        print(line, end="")
        out_lines.append(line)
    proc.wait()
    if proc.returncode != 0:
        sys.exit(f"Command failed: {cmd}")
    return "".join(out_lines)


def parse_avg_ms(output):
    """Extract the number in the final 'avg time / iteration =' line."""
    for line in reversed(output.splitlines()):
        if "avg" in line and "iteration" in line:
            return float(line.strip().split()[-2])  # value before 'ms'
    sys.exit("Could not parse timing from output")


def check_csv(ref, other, label):
    if filecmp.cmp(ref, other, shallow=False):
        print(f"{label}: OK (identical to CPU)")
        return
    sys.exit(f"{label}: CSV differs from CPU reference")


# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Run HW5 experiments (CPU, single GPU, multi GPU)")
    parser.add_argument("maxiter", type=int, help="Jacobi iterations")
    args = parser.parse_args()
    maxiter = args.maxiter

    # sanity check
    for exe in (BIN_CPU, BIN_SINGLE, BIN_MULTI):
        if not Path(exe).exists():
            sys.exit(f"Missing executable {exe}")

    timings = OrderedDict()

    # -------------------------------------------------------------- CPU
    print("\n=== CPU baseline ===")
    out = run(f"{BIN_CPU} {maxiter}")
    timings["CPU"] = parse_avg_ms(out)
    if not Path(CPU_OUT).exists():
        sys.exit("CPU run did not create output_cpu.csv")

    # ------------------------------------------------------- single GPU
    print("\n=== Single-GPU runs ===")
    for T in BLOCK_LIST:
        tag = f"single_T{T}"
        out = run(f"{BIN_SINGLE} {maxiter} {T}")
        timings[tag] = parse_avg_ms(out)
        check_csv(CPU_OUT, SGL_OUT, tag)

    # ------------------------------------------------------- multi GPU
    print("\n=== Multi-GPU runs (all visible GPUs) ===")
    for T in BLOCK_LIST:
        tag = f"multi_T{T}"
        out = run(f"{BIN_MULTI} {maxiter} {T}")
        timings[tag] = parse_avg_ms(out)
        check_csv(CPU_OUT, MULT_OUT, tag)

    # ------------------------------------------------------- summary
    print("\n=== Timing summary (average ms / iteration) ===")
    w = max(len(k) for k in timings)
    best_single = min((v, k) for k, v in timings.items() if k.startswith("single"))[1]
    best_multi  = min((v, k) for k, v in timings.items() if k.startswith("multi"))[1]

    for k, v in timings.items():
        print(f"{k:<{w}} : {v:8.5f}")

    print(f"\nFastest single-GPU config : {best_single}")
    print(f"Fastest multi-GPU  config : {best_multi}")

    # Load CSV
    data = np.loadtxt("output_cpu.csv", delimiter=",")

    # Plot heatmap
    plt.imshow(data, cmap="hot", origin="lower", vmin=273, vmax=400)
    plt.colorbar(label="Temperature (K)")
    plt.title("Heatmap of Steady-State Temperature Distribution")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig("heatmap.png")
    plt.show()

if __name__ == "__main__":
    main()
