import numpy as np
import matplotlib.pyplot as plt

# Load CSV
data = np.loadtxt("output_single.csv", delimiter=",")

# Plot heatmap
plt.imshow(data, cmap="hot", origin="lower", vmin=273, vmax=400)
plt.colorbar(label="Temperature (K)")
plt.title("Heatmap of Steady-State Temperature Distribution")
plt.xlabel("X")
plt.ylabel("Y")
plt.savefig("heatmap.png")
plt.show()