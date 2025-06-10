import subprocess
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import itertools
import re

# Executable names and histogram outputs
EXECUTABLES = {
    "CPU": ["./hw6_cpu"],
    "GPU-Global": "./hw6_gmen",
    "GPU-Shared": "./hw6_shmen"
}
HIST_FILES = {
    "CPU": "cpu_histogram.txt",
    "GPU-Global": "histogram.txt",
    "GPU-Shared": "histogram.txt"
}

# Thread/block configurations to test
THREADS_PER_BLOCK = [128, 256, 512]
BLOCKS_PER_GRID = [64, 128, 256]

results = {}
best_histograms = {}

def parse_time_output(output):
    match = re.search(r"Time to generate\s*[:=]\s*([\d.]+)\s*ms", output)
    return float(match.group(1)) if match else None

def run_command(cmd):
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result.stdout, result.stderr, result.returncode

# Run CPU once
print("\nRunning CPU version...")
stdout, stderr, ret = run_command(EXECUTABLES["CPU"])
if ret != 0:
    print(f"Error running CPU version:\n{stderr}")
else:
    time_ms = parse_time_output(stdout)
    if time_ms is not None:
        print(f"CPU: {time_ms:.2f} ms")
        if os.path.exists(HIST_FILES["CPU"]):
            data = np.loadtxt(HIST_FILES["CPU"])
            x = data[:, 0]
            bin_width = x[1] - x[0]
            y = data[:, 1] / (np.sum(data[:, 1]) * bin_width)
            best_histograms["CPU"] = (x, y)
            results["CPU"] = [("CPU", "-", "-", time_ms)]
    else:
        print("Failed to extract CPU timing.")

# Run GPU versions with block/thread sweep
for version in ["GPU-Global", "GPU-Shared"]:
    exe = EXECUTABLES[version]
    best_time = float("inf")
    best_cfg = None
    best_hist = None
    version_results = []

    print(f"\nSweeping {version}...")
    for tpb, bpg in itertools.product(THREADS_PER_BLOCK, BLOCKS_PER_GRID):
        cmd = [exe, str(tpb), str(bpg)]
        stdout, stderr, ret = run_command(cmd)
        if ret != 0:
            print(f"  Failed: {exe} {tpb} {bpg}")
            continue
        time_ms = parse_time_output(stdout)
        if time_ms is None:
            print(f"  Time not found for {tpb} {bpg}")
            continue

        print(f"  {tpb} threads/block × {bpg} blocks/grid → {time_ms:.2f} ms")
        version_results.append((version, tpb, bpg, time_ms))

        if time_ms < best_time and os.path.exists(HIST_FILES[version]):
            data = np.loadtxt(HIST_FILES[version])
            x = data[:, 0]
            bin_width = x[1] - x[0]
            y = data[:, 1] / (np.sum(data[:, 1]) * bin_width)
            best_time = time_ms
            best_cfg = (tpb, bpg)
            best_hist = (x, y)

    if best_hist is not None:
        best_histograms[version] = best_hist
        results[version] = version_results
        print(f"Best config: {best_cfg[0]} TPB × {best_cfg[1]} BPG → {best_time:.2f} ms")
    else:
        print(f"No valid histogram found for {version}")

# Plot histograms
plt.figure(figsize=(10, 6))
for name, (x, y) in best_histograms.items():
    plt.plot(x, y, label=name)

x_theory = np.linspace(0, 32, 1024)
plt.plot(x_theory, np.exp(-x_theory), 'k--', linewidth=2, label="Theoretical $e^{-x}$")

plt.xlabel("x")
plt.ylabel("Probability Density")
plt.title("Histogram Comparison: CPU vs GPU")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("hw6_histogram_comparison.png", dpi=300)
print("Saved: hw6_histogram_comparison.png")

# Save timing results to CSV
with open("timing_table.csv", "w") as f:
    f.write("Version,ThreadsPerBlock,BlocksPerGrid,Time_ms\n")
    for v in results:
        for version, tpb, bpg, time_ms in results[v]:
            f.write(f"{version},{tpb},{bpg},{time_ms:.2f}\n")
print("Saved: timing_table.csv")

# Summary of best timings
print("\nSummary of best timings:")
for v in results:
    best = min(results[v], key=lambda x: x[3])
    print(f"{best[0]}: {best[3]:.2f} ms (TPB={best[1]}, BPG={best[2]})")
