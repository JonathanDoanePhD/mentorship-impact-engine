"""Simulation-based power for the primary ANCOVA (12-month score ~ program + baseline)."""
import numpy as np
import pandas as pd


def power_grid(n_per_arm=(50, 100, 150, 200, 300, 400, 600), effects=(0.10, 0.20, 0.30),
               baseline_r=0.55, retention=0.78, reps=600, seed=7):
    rng = np.random.default_rng(seed)
    rows = []
    for d in effects:
        for n in n_per_arm:
            hits = 0
            m = int(round(n * retention))  # analyzed per arm after attrition
            for _ in range(reps):
                t = np.r_[np.ones(m), np.zeros(m)]
                y0 = rng.standard_normal(2 * m)
                y1 = baseline_r * y0 + np.sqrt(1 - baseline_r ** 2) * rng.standard_normal(2 * m) + d * t
                X = np.column_stack([np.ones(2 * m), t, y0])
                beta, *_ = np.linalg.lstsq(X, y1, rcond=None)
                resid = y1 - X @ beta
                s2 = resid @ resid / (2 * m - 3)
                se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
                hits += abs(beta[1] / se) > 1.96
            rows.append({"effect_sd": d, "n_per_arm_enrolled": n, "n_per_arm_analyzed": m,
                         "power": hits / reps})
    return pd.DataFrame(rows)


def n_for_power(grid, effect, target=0.8):
    g = grid[grid.effect_sd == effect].sort_values("n_per_arm_enrolled")
    ok = g[g.power >= target]
    return int(ok.n_per_arm_enrolled.iloc[0]) if len(ok) else None
