import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv(r"D:\3ACS\GRM\Projet\images\results\rock_4k.jpg\experiment_results.csv")

# Get unique levels
levels = df["Level"].unique()

# Create plots
fig, axes = plt.subplots(3, 1, figsize=(8, 15))

# Time vs Bandwidth
for level in levels:
    subset = df[df["Level"] == level][df["Factor"] == 4]
    axes[0].plot(subset["Bandwidth"], subset["Time_Banded"], marker="o", label=f"Level {level}")

axes[0].set_xlabel("Bandwidth")
axes[0].set_ylabel("Time (seconds)")
axes[0].set_title("Time vs Bandwidth")
axes[0].legend()
axes[0].grid(True)

# Memory vs Bandwidth
for level in levels:
    subset = df[df["Level"] == level][df["Factor"] == 4]
    axes[1].plot(subset["Bandwidth"], subset["Memory_Banded"], marker="o", label=f"Level {level}")

axes[1].set_xlabel("Bandwidth")
axes[1].set_ylabel("Memory (MB)")
axes[1].set_title("Memory vs Bandwidth")
axes[1].legend()
axes[1].grid(True)

# F1 Score vs Bandwidth
for level in levels:
    subset = df[df["Level"] == level][df["Factor"] == 4]
    axes[2].plot(subset["Bandwidth"], subset["F1_Banded"], marker="o", label=f"Level {level}")

axes[2].set_xlabel("Bandwidth")
axes[2].set_ylabel("DICE Score")
axes[2].set_title("DICE Score vs Bandwidth")
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
fig.savefig(r"D:\3ACS\GRM\Projet\bandwidth_vs_perfs_factor4.png ", format="png")
# plt.show()
