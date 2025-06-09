#!/usr/bin/env python3
import subprocess, sys, time, os, re, math
import numpy as np
import matplotlib.pyplot as plt

# -------- 1.  experiment config -------------------------------------
sizes   = [8, 16, 32, 64]
iters   = {8:800, 16:1500, 32:3500, 64:7500}   # edit as needed

phi_data = {}     # L -> (r, phi_avg)
timings  = {}     # L -> runtime in seconds

# -------- 2.  run solver --------------------------------------------
for L in sizes:
  print(f"Running L={L}  sweeps={iters[L]} ...")
  t0 = time.time()
  proc = subprocess.run(
      ["./hw3", str(L), str(iters[L])],
      capture_output=True, text=True, check=True
  )
  timings[L] = time.time() - t0

  # save raw text for the report
  fname = f"phi_L{L}.dat"
  with open(fname, "w") as f:  f.write(proc.stdout)

  # parse r  phi lines
  r_vals, phi_vals = [], []
  for line in proc.stdout.splitlines():
      if line.startswith("#"): continue
      m = re.match(r"\s*(\d+)\s+([Ee0-9\.\+\-]+)", line)
      if m:
          r_vals.append(int(m.group(1)))
          phi_vals.append(float(m.group(2)))
  phi_data[L] = (np.array(r_vals), np.array(phi_vals))

# -------- 3.  plot φ̄(r) vs 1/r --------------------------------------
plt.figure()
for L,(r,phi) in phi_data.items():
  plt.loglog(r, phi, 'o-', label=f'L={L}')
r_ref = np.arange(1, max(sizes)+1)
plt.loglog(r_ref, 0.0795775/r_ref, 'k--', label='1/(4πr)')
plt.xlabel('r')
plt.ylabel('φ̄(r)')
plt.title('Radial potential vs. Coulomb 1/r')
plt.legend()
plt.grid(True, which='both')
plt.savefig('phi_vs_r.png', dpi=200)
print("Saved plot -> phi_vs_r.png")

# -------- 4.  timing summary ----------------------------------------
print("\nTiming summary:")
for L in sizes:
  print(f"L={L:2d} sweeps={iters[L]:5d}  time={timings[L]:.3f}s")
