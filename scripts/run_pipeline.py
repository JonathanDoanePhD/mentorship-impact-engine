"""Run the full evaluation pipeline.

    python3 -I scripts/run_pipeline.py            # synthetic demo data (default)
    python3 -I scripts/run_pipeline.py --data DIR # real data exported in the same CSV layout

DIR must contain participants.csv, demographics.csv, responses.csv, matches.csv, goals.csv
(see docs/data_schema.md). Real data must stay OUTSIDE this repository.
Outputs: output/results/*.csv, output/results/results.json, output/power/*, output/dashboard.html
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engine import analysis as A  # noqa: E402
from engine.dashboard import build_dashboard  # noqa: E402
from engine.instrument import PRIMARY, SECONDARY  # noqa: E402
from engine.power import power_grid, n_for_power  # noqa: E402
from engine.simulate import simulate, SimConfig  # noqa: E402

TABLES = ["participants", "demographics", "responses", "matches", "goals"]
RES = ROOT / "output" / "results"
POW = ROOT / "output" / "power"
for p in (RES, POW):
    p.mkdir(parents=True, exist_ok=True)

if "--data" in sys.argv:
    src = Path(sys.argv[sys.argv.index("--data") + 1])
    d = {t: pd.read_csv(src / f"{t}.csv") for t in TABLES}
    synthetic = False
else:
    d = simulate(SimConfig())
    syn = ROOT / "data" / "synthetic"
    syn.mkdir(parents=True, exist_ok=True)
    for t in TABLES:
        d[t].to_csv(syn / f"{t}.csv", index=False)
    synthetic = True

f, wide = A.build_frame(d)
ret = A.retention(d, f)
rel = A.reliability(wide)
imp = A.all_impacts(f)
grow = pd.DataFrame([A.growth(f, o) for o in PRIMARY])
sub = A.subgroups(f)
dose = A.dose_response(f)
gas_per, gas_dom = A.gas_tscores(d["goals"])
traj = A.trajectories(f)
bench = pd.read_csv(ROOT / "data" / "benchmarks" / "national_benchmarks.csv")
ib = A.item_benchmarks(d["responses"], d["participants"], bench)

r12 = ret[(ret.wave == 12) & (ret.group == "All")].set_index("arm").retained
base_r = float(imp[imp.wave == 12].baseline_r.mean())
grid = power_grid(baseline_r=base_r, retention=float(r12.mean()))

for name, df in {"retention": ret, "reliability": rel, "impacts": imp, "growth": grow, "subgroups": sub,
                 "dose_response": dose, "gas_tscores": gas_per, "gas_by_domain": gas_dom,
                 "trajectories": traj, "benchmarks_vs_participants": ib}.items():
    df.to_csv(RES / f"{name}.csv", index=False, float_format="%.4f")
grid.to_csv(POW / "power_grid.csv", index=False)

m = d["matches"]
summary = {
    "synthetic": synthetic,
    "enrolled": d["participants"].arm.value_counts().to_dict(),
    "retention_12m": r12.to_dict(),
    "early_closure_rate": float(m.closed_early.mean()),
    "median_months_active": float(m.months_active.median()),
    "baseline_r": base_r,
    "differential_attrition": float(abs(r12["program"] - r12["comparison"])),
    "differential_attrition_flag": bool(abs(r12["program"] - r12["comparison"]) > A.DIFF_ATTRITION_FLAG),
    "low_reliability_scales": sorted(rel[(rel.wave == 0) & (rel.alpha < 0.7)].scale.tolist()),
    "n_for_80pct_power": {str(e): n_for_power(grid, e) for e in (0.1, 0.2, 0.3)},
    "gas_mean_t": float(gas_per.t_score.mean()),
    "gas_share_at_or_above_expected": float((gas_per.t_score >= 50).mean()),
    "primary": PRIMARY, "secondary": SECONDARY,
}
(RES / "results.json").write_text(json.dumps(summary, indent=2))
build_dashboard(ROOT, summary, ret, rel, imp, grow, sub, dose, gas_per, gas_dom, traj, ib, grid)

pd.set_option("display.width", 180)
print(json.dumps(summary, indent=1))
print(imp[["label", "wave", "g", "g_low", "g_high", "g_complete_case", "mde_g", "sensitivity_flag",
           "n_program", "n_comparison"]].round(3).to_string(index=False))
print(grow.round(3).to_string(index=False))
print(rel.pivot(index="scale", columns="wave", values="alpha").round(2).to_string())
print(dose.round(2).to_string(index=False))
print(sub[["label", "dimension", "group", "g", "suppressed"]].round(2).to_string(index=False))
print(ib[["item_id", "participants_share", "benchmark_group", "benchmark_share"]].round(3).to_string(index=False))
