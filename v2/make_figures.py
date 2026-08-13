"""Figures for the v2 manuscript (English labels)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from passport import _pairs, _phases, KERNELS, POWER_KERNELS

ALL = {**POWER_KERNELS, **KERNELS}
rng = np.random.default_rng(7)


def load(nm, n=10000):
    return np.loadtxt(f"data_bench/series/{nm}.csv", skiprows=1)[:n]


# ---------------------------------------------------------------- figure 1
fig, axes = plt.subplots(2, 3, figsize=(13.5, 7))

cases = [
    ("lorenz_x_rho28", "p3", 100, "Lorenz $x$, cubic kernel $S_3$, $K=100$"),
    ("tent_proxy", "p2", 400, "Tent map, quadratic kernel $S_2$, $K=400$"),
    ("tent_proxy", "hash", 200, "Tent map, pseudorandom hash, $K=200$"),
]


def tent(n=10000, mu=1.9999, x0=0.37):
    x = np.empty(n)
    x[0] = x0
    for i in range(1, n):
        x[i] = mu * min(x[i - 1], 1 - x[i - 1])
    return x


SER = {"tent_proxy": tent()}

for ax, (nm, kn, K, title) in zip(axes[0], cases):
    x = SER.get(nm, None)
    if x is None:
        x = load(nm)
    Xi = _phases(x, K, ALL[kn])
    Xin = np.concatenate([_phases(rng.uniform(0, 1, 10000), K, ALL[kn])
                          for _ in range(20)])
    bins = np.linspace(0, 1, 101)
    ax.hist(Xin, bins=bins, density=True, alpha=0.45, color="#999999",
            label="i.i.d. null")
    ax.hist(Xi, bins=bins, density=True, histtype="step", lw=1.6,
            color="#c0392b", label="data")
    for v in (1 / 3, 2 / 3):
        ax.axvline(v, color="k", ls=":", lw=0.7)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(r"$\Xi$")
    ax.set_ylabel("density")
    ax.legend(fontsize=8, frameon=False)

cases2 = [
    ("tent_proxy", "p2", 400, "Quadratic kernel: atoms lie on smooth curves"),
    ("tent_proxy", "p3", 400, "Cubic kernel: lattice at thirds"),
    ("tent_proxy", "hash", 400, "Hash kernel: arithmetic structure destroyed"),
]
for ax, (nm, kn, K, title) in zip(axes[1], cases2):
    x = SER.get(nm, load(nm) if nm not in SER else SER[nm])
    Nc, dN = _pairs(x, K)
    Xi = ALL[kn](Nc, dN)
    ax.scatter(Nc, Xi, s=2, alpha=0.25, c="#2471a3", edgecolors="none")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(r"$N$ (lattice cell)")
    ax.set_ylabel(r"$\Xi$")

plt.tight_layout()
plt.savefig("fig_mechanism.pdf", dpi=200)
plt.savefig("fig_mechanism.png", dpi=140)
print("saved fig_mechanism.pdf/.png")
