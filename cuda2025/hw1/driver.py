import subprocess
import statistics
import re
import matplotlib.pyplot as plt

# Parameters
block_sizes = [64, 128, 256, 512, 1024]
grid_size = 32
num_runs = 5

results = []

for block_size in block_sizes:
    gpu_times = []
    cpu_times = []
    speedups = []

    print(f"Testing threadsPerBlock={block_size}, blocksPerGrid={grid_size}...")

    for _ in range(num_runs):
        try:
            result = subprocess.run(
                ["./hw1", str(block_size), str(grid_size)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )

            gpu_time = cpu_time = speedup = None
            for line in result.stdout.splitlines():
                if "GPU elapsed time" in line:
                    gpu_time = float(re.search(r"([\d.]+)", line).group(1))
                elif "CPU elapsed time" in line:
                    cpu_time = float(re.search(r"([\d.]+)", line).group(1))
                elif "Speed up" in line:
                    speedup = float(re.search(r"([\d.]+)", line).group(1))

            if gpu_time and cpu_time and speedup:
                gpu_times.append(gpu_time)
                cpu_times.append(cpu_time)
                speedups.append(speedup)

        except subprocess.CalledProcessError as e:
            print(f"Error running ./hw1: {e.stderr}")
            break

    if gpu_times:
        results.append({
            "block": block_size,
            "gpu_avg": statistics.mean(gpu_times),
            "gpu_std": statistics.stdev(gpu_times) if len(gpu_times) > 1 else 0,
            "cpu_avg": statistics.mean(cpu_times),
            "speedup_avg": statistics.mean(speedups)
        })

# Print summary table
print("\nSummary:")
print(f"{'Threads/Block':>15} | {'GPU Time (ms)':>13} | {'CPU Time (ms)':>13} | {'Speedup':>8}")
print("-" * 60)
for r in results:
    print(f"{r['block']:>15} | {r['gpu_avg']:>13.3f} | {r['cpu_avg']:>13.3f} | {r['speedup_avg']:>8.2f}")

# Plotting
blocks = [r["block"] for r in results]
gpu_times_avg = [r["gpu_avg"] for r in results]
speedups_avg = [r["speedup_avg"] for r in results]

# GPU time plot
plt.figure(figsize=(8, 5))
plt.plot(blocks, gpu_times_avg, marker='o')
plt.title("GPU Execution Time vs Threads Per Block")
plt.xlabel("Threads Per Block")
plt.ylabel("GPU Time (ms)")
plt.grid(True)
plt.tight_layout()
plt.savefig("gpu_time_vs_threads.png")
plt.show()

# Speedup plot
plt.figure(figsize=(8, 5))
plt.plot(blocks, speedups_avg, marker='o', color='green')
plt.title("Speedup vs Threads Per Block")
plt.xlabel("Threads Per Block")
plt.ylabel("Speedup (CPU Time / GPU Time)")
plt.grid(True)
plt.tight_layout()
plt.savefig("speedup_vs_threads.png")
plt.show()
