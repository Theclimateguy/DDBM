"""Verified chaotic-flow generators.

Every flow ships with a Benettin largest-Lyapunov-exponent check on the ODE
itself (variational method, not on the sampled series), so the ground-truth
label of a generated series is established independently of any detector.
Sampling rate is set from the natural period of the flow: roughly 20-40
samples per orbit, which is what the classical delay-embedding literature
recommends.
"""
import numpy as np


def rk4_step(f, s, dt):
    k1 = f(s); k2 = f(s + dt / 2 * k1)
    k3 = f(s + dt / 2 * k2); k4 = f(s + dt * k3)
    return s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def lyap_benettin(f, jac, s0, dt=0.005, n_steps=200_000, burn=20_000):
    """Largest Lyapunov exponent by Benettin renormalization on the flow."""
    s = np.asarray(s0, float)
    for _ in range(burn):
        s = rk4_step(f, s, dt)
    v = np.random.default_rng(0).normal(size=s.size)
    v /= np.linalg.norm(v)
    acc = 0.0
    for _ in range(n_steps):
        def fv(state):
            x, dv = state[: s.size], state[s.size:]
            return np.concatenate([f(x), jac(x) @ dv])
        st = rk4_step(fv, np.concatenate([s, v]), dt)
        s, v = st[: s.size], st[s.size:]
        nrm = np.linalg.norm(v)
        acc += np.log(nrm)
        v /= nrm
    return acc / (n_steps * dt)


def sample_flow(f, s0, n, dt, sample_every, burn_time=200.0, comp=0):
    s = np.asarray(s0, float)
    for _ in range(int(burn_time / dt)):
        s = rk4_step(f, s, dt)
    out = np.empty(n)
    k = 0
    step = 0
    while k < n:
        s = rk4_step(f, s, dt)
        step += 1
        if step % sample_every == 0:
            out[k] = s[comp]
            k += 1
    return out


# --------------------------------------------------------------- systems
def lorenz_f(sigma=10.0, rho=28.0, beta=8 / 3):
    def f(s):
        x, y, z = s
        return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])

    def jac(s):
        x, y, z = s
        return np.array([[-sigma, sigma, 0.0],
                         [rho - z, -1.0, -x],
                         [y, x, -beta]])
    return f, jac


def rossler_f(a=0.2, b=0.2, c=5.7):
    def f(s):
        x, y, z = s
        return np.array([-y - z, x + a * y, b + z * (x - c)])

    def jac(s):
        x, y, z = s
        return np.array([[0.0, -1.0, -1.0],
                         [1.0, a, 0.0],
                         [z, 0.0, x - c]])
    return f, jac


def chua_f(alpha=15.6, beta=28.0, m0=-8 / 7, m1=-5 / 7):
    """Chua's circuit, dimensionless form, canonical double-scroll."""
    def h(x):
        return m1 * x + 0.5 * (m0 - m1) * (abs(x + 1.0) - abs(x - 1.0))

    def f(s):
        x, y, z = s
        return np.array([alpha * (y - x - h(x)), x - y + z, -beta * y])

    def jac(s):
        x, y, z = s
        dh = m0 if abs(x) < 1.0 else m1
        return np.array([[-alpha * (1.0 + dh), alpha, 0.0],
                         [1.0, -1.0, 1.0],
                         [0.0, -beta, 0.0]])
    return f, jac


# natural periods: Lorenz ~0.75 t.u., Rossler ~6 t.u., Chua ~2.5 t.u.
SPECS = {
    # name: (factory, kwargs, s0, dt, sample_every, comp, expected)
    "lorenz_x_rho28":  (lorenz_f, dict(), [1.0, 1.0, 1.0], 0.002, 15, 0, "Chaos"),
    "lorenz_y_rho28":  (lorenz_f, dict(), [1.0, 1.0, 1.0], 0.002, 15, 1, "Chaos"),
    "lorenz_z_rho28":  (lorenz_f, dict(), [1.0, 1.0, 1.0], 0.002, 15, 2, "Chaos"),
    "lorenz_x_rho10":  (lorenz_f, dict(rho=10.0), [1.0, 1.0, 1.0], 0.002, 15, 0, "Regular"),
    "rossler_x_c5.7":  (rossler_f, dict(), [1.0, 1.0, 0.5], 0.005, 60, 0, "Chaos"),
    "rossler_y_c5.7":  (rossler_f, dict(), [1.0, 1.0, 0.5], 0.005, 60, 1, "Chaos"),
    "rossler_x_c2.5":  (rossler_f, dict(c=2.5), [1.0, 1.0, 0.5], 0.005, 60, 0, "Regular"),
    "chua_x":          (chua_f, dict(), [0.7, 0.0, -0.5], 0.002, 60, 0, "Chaos"),
    "chua_y":          (chua_f, dict(), [0.7, 0.0, -0.5], 0.002, 60, 1, "Chaos"),
}


def generate(name, n=10000, verify=False):
    factory, kw, s0, dt, every, comp, expected = SPECS[name]
    f, jac = factory(**kw)
    x = sample_flow(f, s0, n, dt, every, comp=comp)
    if verify:
        lam = lyap_benettin(f, jac, s0, dt=dt, n_steps=60_000, burn=20_000)
        return x, lam, expected
    return x, None, expected


if __name__ == "__main__":
    print(f"{'system':16s} {'lambda_max':>10s} {'expected':>9s} {'verdict':>9s} {'osc/orbit':>10s}")
    for name in SPECS:
        x, lam, expected = generate(name, n=6000, verify=True)
        verdict = "Chaos" if lam > 0.01 else "Regular"
        # crude oscillations-per-sample check via zero crossings of anomaly
        xc = x - x.mean()
        nz = np.sum(np.diff(np.sign(xc)) != 0)
        per = 2 * len(x) / max(nz, 1)
        flag = "" if verdict == expected else "   <-- MISMATCH"
        print(f"{name:16s} {lam:10.4f} {expected:>9s} {verdict:>9s} {per:10.1f}{flag}")
