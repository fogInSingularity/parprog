# python
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <result.csv>", file=sys.stderr)
    sys.exit(1)

csv_path = sys.argv[1]

try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    print(f"Error: file '{csv_path}' not found", file=sys.stderr)
    sys.exit(1)

t_vals = np.sort(df["t"].unique())
x_vals = np.sort(df["x"].unique())

T_grid, X_grid = np.meshgrid(t_vals, x_vals)
U_grid = df.pivot(index="x", columns="t", values="u").values

fig = plt.figure(figsize=(14, 5))

ax1 = fig.add_subplot(121, projection="3d")
surf = ax1.plot_surface(T_grid, X_grid, U_grid,
                        cmap=cm.viridis, edgecolor="none", alpha=0.9)
fig.colorbar(surf, ax=ax1, shrink=0.5, label="u")
ax1.set_xlabel("t")
ax1.set_ylabel("x")
ax1.set_zlabel("u(x,t)")
ax1.set_title("Solution surface u(x,t)")

ax2 = fig.add_subplot(122)
im = ax2.imshow(
    U_grid,
    origin="lower",
    aspect="auto",
    extent=[t_vals.min(), t_vals.max(), x_vals.min(), x_vals.max()],
    cmap=cm.viridis,
)
fig.colorbar(im, ax=ax2, label="u")
ax2.set_xlabel("t")
ax2.set_ylabel("x")
ax2.set_title("Heatmap u(x,t)")

plt.tight_layout()

out_path = csv_path.rsplit(".", 1)[0] + ".png"
plt.savefig(out_path, dpi=150)
print(f"Saved: {out_path}")
plt.show()
