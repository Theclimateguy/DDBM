"""Cyclotomic kernel family - the generalization implied by the theory.

Why the power kernels work at all: modulo S_p(N) = (N+1)^p - N^p we have
(N+1)^p = N^p, hence rho = (N+1)/N is an exact p-th root of unity in Z/S.
Writing S as a homogeneous form,

    S_p(N) = prod_{d | p, d > 1} Phi_d(N+1, N),

i.e. the modulus is a product of cyclotomic polynomials evaluated at two
CONSECUTIVE integers. That is the whole engine, and it says the power family
is not fundamental: the fundamental object is a single cyclotomic form

    S_n(N) = Phi_n(N+1, N) = N^deg * Phi_n((N+1)/N),

which is irreducible, cheaper (degree phi(n), not n-1), and carries an
n-fold phase lattice. This module builds those kernels and compares them
against the power family.
"""
import numpy as np
import sympy as sp

_x = sp.symbols("x")


def cyclotomic_form(n):
    """Coefficients of the homogenized n-th cyclotomic polynomial Phi_n(N+1, N)
    expanded as a polynomial in N."""
    Nsym = sp.symbols("N")
    phi = sp.cyclotomic_poly(n, _x)
    deg = sp.degree(phi, _x)
    hom = sp.expand(Nsym ** deg * phi.subs(_x, (Nsym + 1) / Nsym))
    poly = sp.Poly(sp.expand(hom), Nsym)
    return [int(c) for c in poly.all_coeffs()], int(deg)


def make_cyclotomic_kernel(n):
    coeffs, deg = cyclotomic_form(n)

    def f(Nc, dN):
        Nc = Nc.astype(object)
        S = np.zeros_like(Nc)
        for c in coeffs:
            S = S * Nc + c
        E = Nc ** deg + dN.astype(object) ** deg
        R = np.array([int(e) % int(s) if s != 0 else 0 for e, s in zip(E, S)],
                     dtype=object)
        return np.array([float(r) / float(s) if s != 0 else 0.0
                         for r, s in zip(R, S)], dtype=float)

    f.__name__ = f"cyc{n}"
    f.degree = deg
    return f


if __name__ == "__main__":
    print("=== cyclotomic forms Phi_n(N+1, N) ===")
    Nsym = sp.symbols("N")
    for n in range(2, 13):
        coeffs, deg = cyclotomic_form(n)
        poly = sum(c * Nsym ** (len(coeffs) - 1 - i) for i, c in enumerate(coeffs))
        print(f"  n={n:2d}  deg={deg}  S = {sp.factor(poly)}")

    print("\n=== lattice denominators of Xi(N, d=0) = (N^deg mod S)/S ===")
    from collections import Counter
    for n in range(2, 13):
        coeffs, deg = cyclotomic_form(n)
        dens = []
        for N in range(60, 400):
            S = 0
            for c in coeffs:
                S = S * N + c
            if S <= 1:
                continue
            xi = (N ** deg % S) / S
            for q in range(1, 30):
                if abs(xi - round(xi * q) / q) < 5.0 / N:
                    dens.append(q)
                    break
            else:
                dens.append(0)
        print(f"  n={n:2d}: {Counter(dens).most_common(2)}")

    # ---------------- power comparison on real systems ----------------
    print("\n=== detection power: cyclotomic vs power kernels ===")
    from passport import Calibration, KERNELS, _ks_D, _phases
    import passport

    CYC = {f"cyc{n}": make_cyclotomic_kernel(n) for n in [3, 5, 6, 7, 8, 12]}
    passport.KERNELS = {**{"p2": KERNELS["p2"], "p3": KERNELS["p3"],
                           "p5": KERNELS["p5"], "hash": KERNELS["hash"]}, **CYC}

    cal = Calibration(10000, b_pool=25, b_cal=150)
    names = ["logistic_r4.00", "henon_x_a1.4", "lorenz_x_rho28", "chua_x",
             "rossler_x_c5.7", "circle_map_qp", "white_noise_1"]
    hdr = list(passport.KERNELS)
    print(f"{'series':18s} " + " ".join(f"{k:>7s}" for k in hdr))
    for nm in names:
        x = np.loadtxt(f"data_bench/series/{nm}.csv", skiprows=1)[:10000]
        r = cal.analyze(x)
        print(f"{nm:18s} " + " ".join(f"{r['profile_z'][k]:7.1f}" for k in hdr),
              flush=True)
