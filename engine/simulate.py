"""Generate a SYNTHETIC evaluation dataset so the pipeline and dashboard can be built and
tested before any real participant exists. No real people are represented.

Assumptions (all adjustable in SimConfig) are drawn from Project 02 and the mentoring
literature (verify before quoting):
  - Program effects at 12 months for a full-length, average-quality match of ~0.20 SD on
    primary outcomes (meta-analytic averages, DuBois et al. 2011; Raposa et al. 2019), with
    larger effects on access outcomes (supportive adults, bridging ties), which the program
    targets directly.
  - Matches that close early (< 6 months) produce no benefit and a small decline
    (Grossman & Rhodes 2002).
  - Compensation for engagement (bigger effect under hardship) and attenuation for hope
    (smaller effect under hardship), as found in Project 02 (NSCH, YRBS).
  - Youth facing hardship start lower on belonging (YRBS gap of ~8 points) and have higher
    attrition.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .instrument import load_items, item_ranges

WAVES = [0, 3, 6, 12]
CONSTRUCT_OF = {"SA": "supportive_adults", "RQ": "relationship_quality", "BE": "belonging",
                "SE": "self_efficacy", "EN": "engagement", "HO": "hope"}
BRIDGING = {"SA3", "SA4", "SA5"}


@dataclass
class SimConfig:
    n_per_arm: int = 400
    sites: int = 4
    site_sd: float = 0.18                 # site random intercept (ICC ~0.03)
    hardship_rate: float = 0.30           # programs recruit higher-need youth
    early_closure_rate: float = 0.28      # matches ending before 6 months
    # 12-month effect (SD units) for a full-length, average-quality match
    effects: dict = field(default_factory=lambda: {
        "supportive_adults": 0.50, "bridging": 0.40, "belonging": 0.25,
        "self_efficacy": 0.20, "engagement": 0.20, "hope": 0.10})
    hardship_modifier: dict = field(default_factory=lambda: {
        "engagement": +0.10, "hope": -0.08})
    early_closure_effect: float = -0.05
    quality_slope: float = 0.10           # extra effect per SD of relationship quality
    retention_12m: dict = field(default_factory=lambda: {"program": 0.82, "comparison": 0.74})
    hardship_retention_penalty: float = 0.08
    item_loading: float = 0.75
    seed: int = 20261113


def _latent_baseline(rng, n, hardship):
    names = ["supportive_adults", "bridging", "belonging", "self_efficacy", "engagement", "hope",
             "relationship_quality"]
    k = len(names)
    corr = np.full((k, k), 0.35); np.fill_diagonal(corr, 1.0)
    z = rng.multivariate_normal(np.zeros(k), corr, size=n)
    lat = pd.DataFrame(z, columns=names)
    shift = {"supportive_adults": -0.40, "bridging": -0.35, "belonging": -0.20,
             "self_efficacy": -0.20, "engagement": -0.25, "hope": -0.30}
    for c, s in shift.items():
        lat[c] += s * hardship
    return lat


# Threshold shift per response set, tuned so baseline answer rates sit near the national
# benchmarks for higher-need youth (e.g. ~75% "usually/always" on engagement items)
SKEW = {"frequency_4": -0.8, "hope_4": -1.0}


def _to_code(latent, lo, hi, rng, loading, response_set=None):
    """Latent (SD units) -> ordinal code via evenly spaced normal thresholds."""
    y = loading * latent + np.sqrt(1 - loading ** 2) * rng.standard_normal(len(latent))
    k = hi - lo + 1
    cuts = np.linspace(-1.4, 1.4, k - 1) - 0.3 + SKEW.get(response_set, 0.0)
    return lo + np.searchsorted(cuts, y)


def simulate(cfg: SimConfig = SimConfig()):
    rng = np.random.default_rng(cfg.seed)
    n = 2 * cfg.n_per_arm
    pid = [f"P{i:04d}" for i in range(1, n + 1)]
    arm = rng.permutation(np.repeat(["program", "comparison"], cfg.n_per_arm))  # lottery
    site = rng.integers(1, cfg.sites + 1, n)
    cohort = rng.choice([2027, 2028], n)
    hardship = rng.binomial(1, cfg.hardship_rate, n)
    participants = pd.DataFrame({"pid": pid, "arm": arm, "site": [f"Site {s}" for s in site],
                                 "cohort": cohort})
    # Restricted, categorical only (kept in a separate file)
    demographics = pd.DataFrame({
        "pid": pid,
        "hardship": hardship,
        "gender": rng.choice(["Girl", "Boy", "Another identity"], n, p=[0.50, 0.46, 0.04]),
        "grade": rng.choice([9, 10, 11, 12], n, p=[0.3, 0.3, 0.25, 0.15]),
        "raceeth": rng.choice(["Black", "Hispanic", "White", "Asian", "Multiracial/other"], n,
                              p=[0.30, 0.32, 0.22, 0.06, 0.10]),
    })
    base = _latent_baseline(rng, n, hardship)
    site_eff = rng.normal(0, cfg.site_sd, cfg.sites + 1)[site]

    # Matches (program arm only)
    prog = arm == "program"
    early = prog & (rng.random(n) < cfg.early_closure_rate + 0.10 * hardship)
    months = np.where(early, rng.integers(1, 6, n), 12)
    quality = base.relationship_quality.values
    matches = pd.DataFrame({
        "match_id": [f"M{i:04d}" for i in range(1, n + 1)], "pid": pid,
        "mentor_id": [f"T{rng.integers(1, 260):03d}" for _ in range(n)],
        "months_active": months, "closed_early": early.astype(int),
        "meetings": np.round(months * rng.uniform(1.4, 2.4, n)).astype(int),
    })[prog].reset_index(drop=True)

    # Retention (monotone dropout)
    p12 = np.where(prog, cfg.retention_12m["program"], cfg.retention_12m["comparison"]) \
        - cfg.hardship_retention_penalty * hardship
    last_wave = np.full(n, 12)
    u = rng.random(n)
    for w, frac in [(3, 0.35), (6, 0.65), (12, 1.0)]:
        drop_by_w = (1 - p12) * frac
        last_wave = np.where((u < drop_by_w) & (last_wave == 12), {3: 0, 6: 3, 12: 6}[w], last_wave)

    items = load_items()
    ranges = item_ranges()
    rows = []
    # Items added after the first release (e.g. AX1) are simulated afterwards with their own
    # random stream, so adding an item never changes the results for existing items.
    LATE_ITEMS = {"AX1"}
    lat_by_wave, alive_by_wave = {}, {}
    for w in WAVES:
        ramp = np.sqrt(w / 12)
        lat = {}
        for c in ["supportive_adults", "bridging", "belonging", "self_efficacy", "engagement", "hope"]:
            eff = cfg.effects[c] + cfg.hardship_modifier.get(c, 0) * hardship + cfg.quality_slope * quality
            eff = np.where(early, cfg.early_closure_effect * (w >= 6), eff * ramp)
            eff = np.where(prog, eff, 0.0)
            drift = 0.03 * (w / 12)  # small maturation for everyone
            lat[c] = base[c].values + site_eff + eff + drift + rng.normal(0, 0.35, n)
        lat["relationship_quality"] = quality
        alive = last_wave >= w
        lat_by_wave[w], alive_by_wave[w] = dict(lat), alive.copy()
        for _, it in items.iterrows():
            if it.item_id in LATE_ITEMS:
                continue
            if it.who == "program" and w == 0:
                continue
            if it.waves == "follow_up" and w == 0:
                continue
            if it.waves in ("baseline_and_12m",) and w not in (0, 12):
                continue
            if it.waves == "baseline" and w != 0:
                continue
            c = "bridging" if it.item_id in BRIDGING else CONSTRUCT_OF.get(it.item_id[:2])
            who = alive.copy()
            if it.who == "program":  # mentor items: only youth with an active match
                who &= prog & ~(early & (w > 3))
            if it.response_set in ("yes_no", "yes_no_unsure") and it.item_id != "HO2":
                # Yes = code 1. Intercepts set baseline rates: SA1 ~85% (NSCH), SA3 ~50%, SA4 ~30%
                a = {"SA1": 1.7, "SA3": 0.0, "SA4": -0.9}.get(it.item_id, 0.0)
                p_yes = 1 / (1 + np.exp(-(a + 1.2 * lat[c])))
                code = np.where(rng.random(n) < p_yes, 1, 2)
                if it.response_set == "yes_no_unsure":
                    code = np.where((code == 2) & (rng.random(n) < 0.3), 3, code)
            elif c is None:  # FU1 / CX1 context items
                if it.item_id == "CX1":
                    code = np.where(hardship == 1, rng.choice([1, 2, 3], n), rng.choice([4, 5], n, p=[0.2, 0.8]))
                else:
                    code = np.clip(np.round(3.4 + 0.6 * base.self_efficacy.values + rng.normal(0, 0.9, n)), 1, 5)
            else:
                lo, hi = ranges[it.item_id]
                latent = lat[c] if it.reverse == 0 else -lat[c]
                code = _to_code(latent, lo, hi, rng, cfg.item_loading, it.response_set)
                if it.item_id == "HO2":  # sadness: yes(1) more likely when hope is low
                    code = np.where(rng.random(n) < 1 / (1 + np.exp(1.0 + 1.1 * lat["hope"])), 1, 2)
            miss = rng.random(n) < 0.02  # item nonresponse
            for p, cd, ok, m in zip(pid, code, who, miss):
                if ok and not m:
                    rows.append((p, w, it.item_id, int(cd)))
    # AX1 (school stress, pilot item): more frequent when self-efficacy and hope are low,
    # following Project 02's PISA finding that anxiety is the largest drag on self-efficacy.
    if "AX1" in set(items.item_id):
        rng_late = np.random.default_rng(cfg.seed + 1)
        lo, hi = ranges["AX1"]
        for w in WAVES:
            lw = lat_by_wave[w]
            code = _to_code(-(0.6 * lw["self_efficacy"] + 0.4 * lw["hope"]), lo, hi, rng_late, cfg.item_loading)
            miss = rng_late.random(n) < 0.02
            for p, cd, ok, m in zip(pid, code, alive_by_wave[w], miss):
                if ok and not m:
                    rows.append((p, w, "AX1", int(cd)))
    responses = pd.DataFrame(rows, columns=["pid", "wave", "item_id", "code"])

    # Goal Attainment Scaling (program youth): 2-3 goals, rated -2..+2 at 6 and 12 months
    goals = []
    domains = ["School", "College & career", "Relationships", "Wellbeing", "Skills"]
    for i in np.where(prog)[0]:
        for gnum in range(1, rng.integers(2, 4) + 1):
            dom = rng.choice(domains, p=[0.34, 0.28, 0.12, 0.12, 0.14])
            for w in (6, 12):
                if last_wave[i] < w:
                    continue
                mu = (-0.6 if early[i] else 0.15 + 0.35 * quality[i]) + (0.2 if w == 12 else 0)
                score = int(np.clip(np.round(rng.normal(mu, 0.95)), -2, 2))
                goals.append((pid[i], f"G{gnum}", dom, w, score))
    goals = pd.DataFrame(goals, columns=["pid", "goal_id", "domain", "wave", "gas_score"])
    return {"participants": participants, "demographics": demographics, "responses": responses,
            "matches": matches, "goals": goals}
