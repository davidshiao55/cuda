#!/usr/bin/env python3
import subprocess, statistics, re, csv, sys, matplotlib.pyplot as plt   # ### NEW/CHG

# ---------- regex patterns -------------------------------------------------
rx_gpu   = re.compile(r"GPU elapsed time:\s*([\d.]+)")
rx_cpu   = re.compile(r"CPU elapsed time:\s*([\d.]+)")
rx_speed = re.compile(r"Speed up\s*=\s*([\d.]+)")

def run_once(threads, grids, gpus):
    """Launch ./hw4 and return (gpu_ms, cpu_ms, speedup) or None on failure."""
    try:
        out = subprocess.check_output(
            ["./hw4", str(gpus), str(threads), str(grids)],
            stderr=subprocess.STDOUT, text=True
        )
    except subprocess.CalledProcessError as e:
        print(f"[!] hw4 exited with error for TPB={threads} GRID={grids}:\n{e.output}")
        return None

    try:
        gpu   = float(rx_gpu.search(out).group(1))
        cpu   = float(rx_cpu.search(out).group(1))
        spdup = float(rx_speed.search(out).group(1))
        return gpu, cpu, spdup
    except AttributeError:
        print(f"[!] Could not parse output for TPB={threads} GRID={grids}")
        return None

# ---------- parameter sweep lists -----------------------------------------
grid_sizes  = [64, 128, 256, 512, 1024]
block_sizes = [64, 128, 256, 512, 1024]
num_runs    = 5
gpus        = 2                                                     # ### NEW/CHG

# ---------- sweep ----------------------------------------------------------
results = []
for blocks in block_sizes:
    for grids in grid_sizes:
        gpu_t, cpu_t, spdup = [], [], []
        tag = f"TPB={blocks:4d}, GRID={grids:4d}"
        print(f"· {tag:<25} … ", end="", flush=True)

        for _ in range(num_runs):
            triple = run_once(blocks, grids, gpus)                  # ### NEW/CHG
            if triple is None:
                break
            g, c, s = triple
            gpu_t.append(g); cpu_t.append(c); spdup.append(s)

        if len(gpu_t) == num_runs:
            print("ok")
            results.append({
                "tpb": blocks,
                "grid": grids,
                "gpu_mean": statistics.mean(gpu_t),
                "cpu_mean": statistics.mean(cpu_t),
                "speed_mean": statistics.mean(spdup)
            })
        else:
            print("failed – skipped")

if not results:
    print("No successful runs – nothing to plot.")
    sys.exit(1)

# ---------- table on stdout ------------------------------------------------
print(f"\nSummary ({num_runs} runs each):")
print(f"{'TPB':>6} {'GRID':>6} | {'GPU ms':>10} | {'CPU ms':>10} | {'Speed-up':>8}")
print("-"*60)
for r in results:
    print(f"{r['tpb']:6d} {r['grid']:6d} | " f"{r['gpu_mean']:10.3f} | " f"{r['cpu_mean']:10.3f} | {r['speed_mean']:8.2f}")

# ---------- CSV ------------------------------------------------------------
with open("hw4_sweep.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["threadsPerBlock","blocksPerGrid","gpu_ms","cpu_ms","speedup"])
    for r in results:
        w.writerow([r['tpb'], r['grid'], r['gpu_mean'], r['cpu_mean'], r['speed_mean']])
print("\n[+] Results written to hw4_sweep.csv")

# ---------- plots ----------------------------------------------------------
unique_grids = sorted({r['grid'] for r in results})
colours = plt.cm.tab10.colors

# GPU time plot
plt.figure(figsize=(7,4))
for idx,g in enumerate(unique_grids):
    xs = [r['tpb']      for r in results if r['grid'] == g]
    ys = [r['gpu_mean'] for r in results if r['grid'] == g]
    plt.plot(xs, ys, marker='o', label=f"GRID={g}", color=colours[idx % 10])
plt.title("GPU execution time vs Threads-per-Block")
plt.xlabel("Threads per block")
plt.ylabel("GPU time (ms, lower is better)")
plt.grid(True); plt.legend(); plt.tight_layout()
plt.savefig("gpu_time_vs_tpb.png")
print("[+] gpu_time_vs_tpb.png saved")

# Speed-up plot
plt.figure(figsize=(7,4))
for idx,g in enumerate(unique_grids):
    xs = [r['tpb']        for r in results if r['grid'] == g]
    ys = [r['speed_mean'] for r in results if r['grid'] == g]
    plt.plot(xs, ys, marker='o', label=f"GRID={g}", color=colours[idx % 10])
plt.title("Speed-up (CPU / GPU)")
plt.xlabel("Threads per block")
plt.ylabel("Speed-up (×)")
plt.grid(True); plt.legend(); plt.tight_layout()
plt.savefig("speedup_vs_tpb.png")
print("[+] speedup_vs_tpb.png saved")

plt.show()
