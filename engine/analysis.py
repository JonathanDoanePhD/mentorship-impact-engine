"""Evaluation analyses. Every estimate comes with its uncertainty; small cells are suppressed."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from .instrument import SCALES, PRIMARY, SECONDARY, LABELS, score_items, score_scales, cronbach_alpha

MIN_CELL = 10           # suppress any reported figure based on fewer youth than this
MIN_SUBGROUP = 20       # subgroup impact estimates need this many youth per arm (precision + privacy)
DIFF_ATTRITION_FLAG = 0.07  # flag if 12-month retention differs by more than this between arms


def build_frame(d):
    wide_items = score_items(d["responses"])
    scales = score_scales(wide_items)
    f = (scales.merge(d["participants"], on="pid").merge(d["demographics"], on="pid", how="left")
         .merge(d["matches"][["pid", "months_active", "closed_early", "meetings"]], on="pid", how="left"))
    f["program"] = (f.arm == "program").astype(int)
    return f, wide_items


def retention(d, f):
    base = d["participants"].merge(d["demographics"][["pid", "hardship"]], on="pid")
    out = []
    for w in [0, 3, 6, 12]:
        present = set(f[(f.wave == w) & f.belonging.notna()].pid)
        for arm in ["program", "comparison"]:
            b = base[base.arm == arm]
            out.append({"wave": w, "arm": arm, "group": "All", "n": len(b),
                        "retained": float(b.pid.isin(present).mean())})
            for h, lab in [(1, "Hardship"), (0, "No hardship")]:
                bb = b[b.hardship == h]
                out.append({"wave": w, "arm": arm, "group": lab, "n": len(bb),
                            "retained": float(bb.pid.isin(present).mean()) if len(bb) >= MIN_CELL else None})
    return pd.DataFrame(out)


def reliability(wide_items):
    rows = []
    for s, its in SCALES.items():
        if len(its) < 2:
            continue
        for w in sorted(wide_items.wave.unique()):
            sub = wide_items[wide_items.wave == w][[i for i in its if i in wide_items]]
            if sub.dropna().shape[0] >= 30:
                rows.append({"scale": s, "wave": w, "alpha": cronbach_alpha(sub), "n": int(sub.dropna().shape[0])})
    return pd.DataFrame(rows)


def _hedges_g(coef, se, sd, n1, n0):
    j = 1 - 3 / (4 * (n1 + n0) - 9)
    return coef / sd * j, se / sd * j


def impact(f, outcome, wave=12, subset=None, min_n=MIN_CELL):
    """ANCOVA of follow-up score on program + baseline score + site.
    Returns complete-case and inverse-probability-of-retention-weighted (IPW) estimates."""
    b = f[f.wave == 0][["pid", outcome, "program", "site", "hardship"]].rename(columns={outcome: "y0"})
    fu = f[f.wave == wave][["pid", outcome]].rename(columns={outcome: "y1"})
    df = b.merge(fu, on="pid", how="left")
    if subset is not None:
        df = df[subset(df)]
    df = df.dropna(subset=["y0"])
    df["retained"] = df.y1.notna().astype(int)
    n1, n0 = int(((df.program == 1) & (df.retained == 1)).sum()), int(((df.program == 0) & (df.retained == 1)).sum())
    if min(n1, n0) < min_n:
        return None
    sd = df.y0.std(ddof=1)
    # Retention model -> weights
    Xr = sm.add_constant(pd.concat([df[["program", "y0", "hardship"]],
                                    pd.get_dummies(df.site, drop_first=True, dtype=float)], axis=1))
    try:
        pr = sm.Logit(df.retained, Xr).fit(disp=0).predict(Xr)
    except Exception:
        pr = pd.Series(df.retained.mean(), index=df.index)
    df["ipw"] = 1 / pr.clip(lower=0.05)
    r = df[df.retained == 1]
    X = sm.add_constant(pd.concat([r[["program", "y0"]], pd.get_dummies(r.site, drop_first=True, dtype=float)], axis=1))
    cc = sm.OLS(r.y1, X).fit(cov_type="HC1")
    ip = sm.WLS(r.y1, X, weights=r.ipw).fit(cov_type="HC1")
    g_cc, se_cc = _hedges_g(cc.params["program"], cc.bse["program"], sd, n1, n0)
    g_ip, se_ip = _hedges_g(ip.params["program"], ip.bse["program"], sd, n1, n0)
    r_base = np.corrcoef(r.y0, r.y1)[0, 1]
    mde = 2.8 * np.sqrt((1 - r_base ** 2) * (1 / n1 + 1 / n0))  # 80% power, two-sided 5%
    return {"outcome": outcome, "label": LABELS[outcome], "wave": wave,
            "role": "primary" if outcome in PRIMARY else "secondary",
            "diff_points": ip.params["program"], "g": g_ip, "g_low": g_ip - 1.96 * se_ip, "g_high": g_ip + 1.96 * se_ip,
            "g_complete_case": g_cc, "g_cc_low": g_cc - 1.96 * se_cc, "g_cc_high": g_cc + 1.96 * se_cc,
            "p_value": float(ip.pvalues["program"]), "n_program": n1, "n_comparison": n0,
            "mde_g": mde, "baseline_r": r_base,
            "sensitivity_flag": bool(abs(g_ip - g_cc) > 0.05)}


def all_impacts(f):
    rows = [impact(f, o, w) for o in PRIMARY + SECONDARY for w in (6, 12)]
    return pd.DataFrame([r for r in rows if r])


def growth(f, outcome):
    """Mixed model: score ~ months x program, random intercept per youth (secondary analysis)."""
    df = f[["pid", "wave", "program", outcome]].dropna()
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = smf.mixedlm(f"{outcome} ~ wave * program", df, groups=df.pid).fit(reml=True)
    k = "wave:program"
    return {"outcome": outcome, "slope_diff_per_month": m.params[k],
            "low": m.conf_int().loc[k, 0], "high": m.conf_int().loc[k, 1]}


def subgroups(f, outcomes=PRIMARY, wave=12):
    groups = {
        "Household hardship": [("Yes", lambda d: d.hardship == 1), ("No", lambda d: d.hardship == 0)],
    }
    demo = f[f.wave == 0][["pid", "gender", "raceeth"]]
    rows = []
    for o in outcomes:
        for gname, levels in groups.items():
            for lab, fn in levels:
                r = impact(f, o, wave, subset=fn, min_n=MIN_SUBGROUP)
                rows.append({"outcome": o, "label": LABELS[o], "dimension": gname, "group": lab,
                             **({"g": r["g"], "g_low": r["g_low"], "g_high": r["g_high"],
                                 "n_program": r["n_program"], "n_comparison": r["n_comparison"]}
                                if r else {"g": None, "suppressed": True})})
        for dim, col in [("Gender", "gender"), ("Race/ethnicity", "raceeth")]:
            for lab in sorted(demo[col].dropna().unique()):
                ids = set(demo[demo[col] == lab].pid)
                r = impact(f, o, wave, subset=lambda d, ids=ids: d.pid.isin(ids), min_n=MIN_SUBGROUP)
                rows.append({"outcome": o, "label": LABELS[o], "dimension": dim, "group": lab,
                             **({"g": r["g"], "g_low": r["g_low"], "g_high": r["g_high"],
                                 "n_program": r["n_program"], "n_comparison": r["n_comparison"]}
                                if r else {"g": None, "suppressed": True})})
    out = pd.DataFrame(rows)
    out["suppressed"] = out.g.isna()
    return out


def dose_response(f, outcome="belonging"):
    """Program youth only: 12-month change by match status (observational)."""
    p = f[f.program == 1]
    b = p[p.wave == 0][["pid", outcome]].rename(columns={outcome: "y0"})
    e = p[p.wave == 12][["pid", outcome, "closed_early", "months_active"]].rename(columns={outcome: "y1"})
    q = p[p.wave == 3][["pid", "relationship_quality"]]
    df = b.merge(e, on="pid").merge(q, on="pid", how="left")
    df["change"] = df.y1 - df.y0
    out = []
    for lab, sel in [("Match lasted 12 months", df.closed_early == 0), ("Closed early (< 6 months)", df.closed_early == 1)]:
        s = df[sel].change.dropna()
        if len(s) >= MIN_CELL:
            out.append({"group": lab, "n": len(s), "mean_change": s.mean(),
                        "low": s.mean() - 1.96 * s.std(ddof=1) / np.sqrt(len(s)),
                        "high": s.mean() + 1.96 * s.std(ddof=1) / np.sqrt(len(s))})
    full = df[df.closed_early == 0].dropna(subset=["relationship_quality"])
    if len(full) >= 30:
        terc = pd.qcut(full.relationship_quality, 3, labels=["Lower", "Middle", "Higher"])
        for lab in ["Lower", "Middle", "Higher"]:
            s = full[terc == lab].change.dropna()
            out.append({"group": f"{lab} relationship quality (3 mo)", "n": len(s), "mean_change": s.mean(),
                        "low": s.mean() - 1.96 * s.std(ddof=1) / np.sqrt(len(s)),
                        "high": s.mean() + 1.96 * s.std(ddof=1) / np.sqrt(len(s))})
    comp = f[f.program == 0]
    cb = comp[comp.wave == 0][["pid", outcome]].merge(comp[comp.wave == 12][["pid", outcome]], on="pid", suffixes=("0", "1"))
    s = (cb[outcome + "1"] - cb[outcome + "0"]).dropna()
    out.append({"group": "Comparison group", "n": len(s), "mean_change": s.mean(),
                "low": s.mean() - 1.96 * s.std(ddof=1) / np.sqrt(len(s)),
                "high": s.mean() + 1.96 * s.std(ddof=1) / np.sqrt(len(s))})
    return pd.DataFrame(out)


def gas_tscores(goals, wave=12, rho=0.3):
    """Kiresuk & Sherman (1968) summary T-score per youth, equal goal weights."""
    g = goals[goals.wave == wave]
    def t(x):
        k = len(x)
        return 50 + 10 * x.sum() / np.sqrt((1 - rho) * k + rho * k ** 2)
    per = g.groupby("pid").gas_score.apply(t).rename("t_score").reset_index()
    by_domain = g.groupby("domain").agg(goals=("gas_score", "size"),
                                         met_or_exceeded=("gas_score", lambda s: (s >= 0).mean())).reset_index()
    return per, by_domain


def trajectories(f, outcomes=PRIMARY + SECONDARY):
    rows = []
    for o in outcomes:
        for (w, arm), s in f.groupby(["wave", "arm"])[o]:
            s = s.dropna()
            if len(s) >= MIN_CELL:
                rows.append({"outcome": o, "label": LABELS[o], "wave": w, "arm": arm, "n": len(s), "mean": s.mean(),
                             "low": s.mean() - 1.96 * s.std(ddof=1) / np.sqrt(len(s)),
                             "high": s.mean() + 1.96 * s.std(ddof=1) / np.sqrt(len(s))})
    return pd.DataFrame(rows)


def item_benchmarks(responses, participants, benchmarks):
    """Baseline share of participants giving the benchmark-defined answer vs. national figures."""
    defs = {  # item -> codes counted as "yes" (participant coding in instrument/response_sets.csv)
        "SA1": [1], "BE1": [4, 5], "EN1": [3, 4], "EN2": [3, 4], "EN3": [3, 4], "EN4": [3, 4],
        "EN5": [3, 4], "HO1": [3, 4], "HO2": [1],
    }
    b = responses[responses.wave == 0]
    rows = []
    for item, codes in defs.items():
        s = b[b.item_id == item].code
        if len(s) < MIN_CELL:
            continue
        p = s.isin(codes).mean()
        se = np.sqrt(p * (1 - p) / len(s))
        nat = benchmarks[benchmarks.item_id == item]
        for _, nb in nat.iterrows():
            rows.append({"item_id": item, "participants_share": p, "low": p - 1.96 * se, "high": p + 1.96 * se,
                         "n": len(s), "benchmark_group": nb.group, "benchmark_share": nb.share,
                         "benchmark_low": nb.low, "benchmark_high": nb.high, "benchmark_source": nb.source,
                         "benchmark_note": nb.note, "measure": nb.measure})
    return pd.DataFrame(rows)
