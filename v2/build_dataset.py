"""Build the clean benchmark dataset: real data + verified synthetic systems.

Writes data_bench/series/*.csv (one column 'value') plus data_bench/manifest.csv
with ground-truth labels and provenance. Everything downstream reads only the
manifest, so a rerun is fully reproducible from these files.

Ground-truth policy
-------------------
Synthetic systems: label established by Benettin largest Lyapunov exponent on
the ODE/map itself (flows.py), never by any detector.
Real data: label is the literature consensus where one exists, and "unknown"
otherwise. Unknown-label series are reported, not scored.
"""
import csv
import glob
import os

import numpy as np

import flows

RAW = "data_bench/raw"
OUT = "data_bench/series"
os.makedirs(OUT, exist_ok=True)

RNG = np.random.default_rng(20260805)
N = 10000
rows = []


def emit(name, x, label, group, source, note=""):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    with open(f"{OUT}/{name}.csv", "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["value"])
        w.writerows([[f"{v:.10g}"] for v in x])
    rows.append(dict(name=name, label=label, group=group, n=len(x),
                     source=source, note=note))
    print(f"  {name:28s} n={len(x):7d} label={label:8s} {group}")


# ------------------------------------------------------------ 1. synthetic
print("== synthetic maps ==")


def logistic(r, n, x0=0.1, burn=1000):
    x = x0
    for _ in range(burn):
        x = r * x * (1 - x)
    out = np.empty(n)
    for i in range(n):
        x = r * x * (1 - x)
        out[i] = x
    return out


def logistic_lyap(r, n=200000, x0=0.1):
    x, acc = x0, 0.0
    for _ in range(1000):
        x = r * x * (1 - x)
    for _ in range(n):
        acc += np.log(abs(r * (1 - 2 * x)) + 1e-300)
        x = r * x * (1 - x)
    return acc / n


def henon(n, a=1.4, b=0.3, comp="x", burn=1000):
    x, y = 0.1, 0.1
    for _ in range(burn):
        x, y = 1 - a * x * x + y, b * x
    out = np.empty(n)
    for i in range(n):
        x, y = 1 - a * x * x + y, b * x
        out[i] = x if comp == "x" else y
    return out


def henon_lyap(a=1.4, b=0.3, n=200000):
    x, y = 0.1, 0.1
    v = np.array([1.0, 0.0])
    acc = 0.0
    for _ in range(1000):
        x, y = 1 - a * x * x + y, b * x
    for _ in range(n):
        J = np.array([[-2 * a * x, 1.0], [b, 0.0]])
        v = J @ v
        nr = np.linalg.norm(v)
        acc += np.log(nr)
        v /= nr
        x, y = 1 - a * x * x + y, b * x
    return acc / n


for r in [2.80, 3.00, 3.50, 3.57, 3.70, 3.90, 4.00]:
    lam = logistic_lyap(r)
    lab = "Chaos" if lam > 0.005 else "Regular"
    emit(f"logistic_r{r:.2f}", logistic(r, N), lab, "map", "synthetic",
         f"lyap={lam:.4f}")

for a, comp in [(1.4, "x"), (1.4, "y"), (1.1, "x"), (1.3, "x")]:
    lam = henon_lyap(a=a)
    lab = "Chaos" if lam > 0.005 else "Regular"
    emit(f"henon_{comp}_a{a}", henon(N, a=a, comp=comp), lab, "map", "synthetic",
         f"lyap={lam:.4f}")

print("== synthetic flows (Benettin-verified) ==")
for name in flows.SPECS:
    x, lam, expected = flows.generate(name, n=N, verify=True)
    lab = "Chaos" if lam > 0.01 else "Regular"
    if lab != expected:
        print(f"    ! {name}: expected {expected}, Lyapunov says {lab} (lam={lam:.4f})"
              f" -- using Lyapunov")
    emit(name, x, lab, "flow", "synthetic", f"lyap={lam:.4f}")

print("== stochastic controls ==")
from scipy.signal import lfilter

for s in [1, 2, 3]:
    emit(f"white_noise_{s}", np.random.default_rng(s).normal(size=N), "Noise",
         "stochastic", "synthetic")
emit("white_uniform", RNG.uniform(size=N), "Noise", "stochastic", "synthetic")
emit("student_t3", RNG.standard_t(3, size=N), "Noise", "stochastic", "synthetic")
for phi in [0.3, 0.7, 0.9]:
    emit(f"ar1_phi{phi}", lfilter([1.0], [1.0, -phi], RNG.normal(size=N + 500))[500:],
         "Noise", "stochastic", "synthetic")
emit("arma22", lfilter([1.0, 0.4, -0.3], [1.0, -0.6, 0.2], RNG.normal(size=N + 500))[500:],
     "Noise", "stochastic", "synthetic")
emit("random_walk", np.cumsum(np.random.default_rng(4).normal(size=N)), "Noise",
     "stochastic", "synthetic")


def garch(n, rng, out="ret"):
    x = np.zeros(n); sig = np.zeros(n)
    s2 = 0.01 / (1 - 0.05 - 0.94)
    for i in range(n):
        sig[i] = np.sqrt(s2)
        x[i] = sig[i] * rng.normal()
        s2 = 0.01 + 0.05 * x[i] ** 2 + 0.94 * s2
    return x if out == "ret" else sig


emit("garch_returns", garch(N, RNG, "ret"), "Noise", "stochastic", "synthetic")
emit("garch_volatility", garch(N, RNG, "vol"), "Noise", "stochastic", "synthetic")
emit("fgn_h0.8", np.cumsum(RNG.normal(size=N)) / np.sqrt(np.arange(1, N + 1)) ** 0.2,
     "Noise", "stochastic", "synthetic", "approx long-memory")

print("== periodic / quasi-periodic ==")
t = np.arange(N)
emit("sine_pure", np.sin(2 * np.pi * 0.05 * t), "Regular", "periodic", "synthetic")
emit("sine_plus_noise", np.sin(2 * np.pi * 0.05 * t) + 0.3 * RNG.normal(size=N),
     "Regular", "periodic", "synthetic")
emit("quasi_sine_sum",
     np.sin(2 * np.pi * 0.05 * t) + 0.5 * np.sin(2 * np.pi * 0.05 * np.sqrt(2) * t),
     "Regular", "periodic", "synthetic")
emit("circle_map_qp", np.sin(2 * np.pi * np.mod(t * (np.sqrt(5) - 1) / 2, 1.0)),
     "Regular", "periodic", "synthetic")

print("== chaos + noise (SNR ladder) ==")


def with_snr(x, snr_db, rng):
    x = (x - x.mean()) / x.std()
    return x + 10 ** (-snr_db / 20) * rng.normal(size=x.size)


base_h = henon(N)
base_l = logistic(4.0, N)
for db in [5, 10, 20]:
    emit(f"henon_SNR{db}dB", with_snr(base_h, db, RNG), "Mixed", "mixed", "synthetic")
    emit(f"logistic_SNR{db}dB", with_snr(base_l, db, RNG), "Mixed", "mixed", "synthetic")

print("== surrogates (nonlinearity removed -> must read as Noise) ==")


def iaaft(x, rng, n_iter=200):
    x = np.asarray(x, float)
    amp = np.abs(np.fft.rfft(x))
    xs = np.sort(x)
    y = rng.permutation(x)
    for _ in range(n_iter):
        Y = np.fft.rfft(y)
        y = np.fft.irfft(amp * np.exp(1j * np.angle(Y)), n=x.size)
        y = xs[np.argsort(np.argsort(y))]
    return y


lor, _, _ = flows.generate("lorenz_x_rho28", n=N)
emit("iaaft_lorenz_x", iaaft(lor, RNG), "Noise", "surrogate", "synthetic")
emit("iaaft_henon_x", iaaft(base_h, RNG), "Noise", "surrogate", "synthetic")
emit("shuffle_lorenz_x", RNG.permutation(lor), "Noise", "surrogate", "synthetic")

# ------------------------------------------------------------ 2. real data
print("== real: finance ==")
sp = list(csv.reader(open(f"{RAW}/spx_daily.csv")))[1:]
dates = [r[0] for r in sp]
close = np.array([float(r[1]) for r in sp])
logret = np.diff(np.log(close))
emit("sp500_log_returns_full", logret, "unknown", "finance",
     f"Yahoo ^GSPC {dates[0]}..{dates[-1]}", "efficient-market prior: Noise")
i0 = next(i for i, d in enumerate(dates) if d >= "2010-01-01")
i1 = next(i for i, d in enumerate(dates) if d >= "2025-01-01")
emit("sp500_log_returns_2010_2024", np.diff(np.log(close[i0:i1])), "unknown",
     "finance", "Yahoo ^GSPC 2010-2024", "matches original DDBM paper window")
w = 22
rv = np.array([np.std(logret[i:i + w]) for i in range(len(logret) - w)])
emit("sp500_realized_vol", rv, "unknown", "finance", "Yahoo ^GSPC 22d rolling std")

print("== real: climate ==")
# HadCET daily, tenths of degC, rows = year, day-of-month, 12 monthly columns
cet_vals = {}
for line in open(f"{RAW}/cet_daily.dat"):
    p = line.split()
    if len(p) < 14:
        continue
    yr, day = int(p[0]), int(p[1])
    for mon, v in enumerate(p[2:14], start=1):
        v = int(v)
        if v > -900:
            cet_vals[(yr, mon, day)] = v / 10.0
keys = sorted(cet_vals)
cet = np.array([cet_vals[k] for k in keys])
doy_idx = np.array([(k[1] - 1) * 31 + (k[2] - 1) for k in keys])
clim = np.zeros(372)
for b in range(372):
    m = doy_idx == b
    if m.any():
        clim[b] = cet[m].mean()
emit("cet_daily_raw", cet, "unknown", "climate",
     f"HadCET {keys[0][0]}-{keys[-1][0]} daily")
emit("cet_daily_anomaly", cet - clim[doy_idx], "unknown", "climate",
     "HadCET daily, seasonal cycle removed", "Hasselmann prior: Noise")

nao = []
for line in open(f"{RAW}/nao_daily.txt"):
    p = line.split()
    if len(p) == 4:
        try:
            nao.append(float(p[3]))
        except ValueError:
            pass
emit("nao_daily", np.asarray(nao), "unknown", "climate",
     "NOAA CPC daily NAO 1950-2026")


def read_psl(path):
    lines = [l.rstrip("\n") for l in open(path)]
    vals = []
    for line in lines[1:]:
        p = line.split()
        if len(p) != 13:
            continue
        try:
            int(p[0])
        except ValueError:
            continue
        for v in p[1:]:
            v = float(v)
            if v > -99:
                vals.append(v)
    return np.asarray(vals)


for nm, fn in [("nino34_monthly", "nina34_monthly.txt"),
               ("pdo_monthly", "pdo_monthly.txt"),
               ("amo_monthly", "amo_monthly.txt")]:
    v = read_psl(f"{RAW}/{fn}")
    emit(nm, v, "unknown", "climate", f"NOAA PSL {fn}", "n<1000: low power")

print("== real: physiology ==")
for rec in sorted(glob.glob(f"{RAW}/rr_*.txt")):
    nm = os.path.basename(rec)[3:-4]
    rr = np.loadtxt(rec)
    grp = "healthy" if nm.startswith("nsr") else "CHF"
    emit(f"rr_{nm}", rr, "unknown", "physiology",
         f"PhysioNet {'nsr2db' if grp == 'healthy' else 'chf2db'} {nm}",
         f"HRV {grp}")

bonn_sets = {"setA": "healthy_eyes_open", "setB": "healthy_eyes_closed",
             "setC": "interictal_opposite", "setD": "interictal_focus",
             "setE": "seizure"}
for st, desc in bonn_sets.items():
    files = sorted(glob.glob(f"{RAW}/bonn/Datasets/{st}/*.txt")
                   + glob.glob(f"{RAW}/bonn/Datasets/{st}/*.TXT"))
    for i, f in enumerate(files[:20]):  # 20 segments per class
        x = np.loadtxt(f)
        emit(f"eeg_{st}_{i:02d}", x, "unknown", "eeg",
             f"Bonn {st} {os.path.basename(f)}", desc)

with open("data_bench/manifest.csv", "w", newline="") as fp:
    w = csv.DictWriter(fp, fieldnames=["name", "label", "group", "n", "source", "note"])
    w.writeheader()
    w.writerows(rows)

print(f"\nwrote {len(rows)} series -> data_bench/manifest.csv")
import collections
print(dict(collections.Counter(r["group"] for r in rows)))
print(dict(collections.Counter(r["label"] for r in rows)))
