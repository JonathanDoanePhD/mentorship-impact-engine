"""Load the survey instrument (items + response sets) and score responses."""
from pathlib import Path

import numpy as np
import pandas as pd

INSTRUMENT = Path(__file__).resolve().parents[1] / "instrument"

# Scales used as outcomes (item ids). Program-only scales are scored only where answered.
SCALES = {
    "supportive_adults": ["SA1", "SA2"],
    "bridging": ["SA3", "SA4", "SA5"],
    "relationship_quality": ["RQ1", "RQ2", "RQ3", "RQ4", "RQ5"],
    "belonging": ["BE1", "BE2", "BE3", "BE4"],
    "self_efficacy": ["SE1", "SE2", "SE3", "SE4", "SE5", "SE6"],
    "engagement": ["EN1", "EN2", "EN3", "EN4", "EN5"],
    "hope": ["HO1"],
}
PRIMARY = ["belonging", "self_efficacy", "engagement"]
SECONDARY = ["supportive_adults", "bridging", "hope"]
LABELS = {
    "supportive_adults": "Supportive adults", "bridging": "Bridging ties",
    "relationship_quality": "Mentor relationship quality", "belonging": "School belonging",
    "self_efficacy": "Self-efficacy", "engagement": "Engagement & persistence", "hope": "Hope",
}


def load_items():
    return pd.read_csv(INSTRUMENT / "items.csv")


def load_response_sets():
    return pd.read_csv(INSTRUMENT / "response_sets.csv")


def item_ranges():
    """(min_code, max_code) for each item's response set, ignoring 'not sure'-type blanks."""
    items, rs = load_items(), load_response_sets()
    out = {}
    for _, it in items.iterrows():
        codes = rs[(rs.response_set == it.response_set) & rs.scored_value.notna()].code
        out[it.item_id] = (int(codes.min()), int(codes.max()))
    return out


def score_items(responses):
    """Long responses (pid, wave, item_id, code) -> wide item scores on a 0-1 scale
    (reverse-coded items flipped), so items with different response sets average sensibly."""
    items, rs = load_items(), load_response_sets()
    val = rs.set_index(["response_set", "code"]).scored_value
    r = responses.merge(items[["item_id", "response_set", "reverse"]], on="item_id")
    r["value"] = [val.get((s, c), np.nan) for s, c in zip(r.response_set, r.code)]
    lo = r.groupby("item_id").response_set.first().map(
        lambda s: rs[(rs.response_set == s)].scored_value.min())
    hi = r.groupby("item_id").response_set.first().map(
        lambda s: rs[(rs.response_set == s)].scored_value.max())
    r["value01"] = (r.value - r.item_id.map(lo)) / (r.item_id.map(hi) - r.item_id.map(lo))
    r.loc[r.reverse == 1, "value01"] = 1 - r.loc[r.reverse == 1, "value01"]
    return r.pivot_table(index=["pid", "wave"], columns="item_id", values="value01").reset_index()


def score_scales(wide, min_items=0.5):
    """Scale score = mean of available items (0-1) x 100, if at least half the items answered."""
    out = wide[["pid", "wave"]].copy()
    for s, its in SCALES.items():
        its = [i for i in its if i in wide]
        if not its:
            continue
        m = wide[its]
        ok = m.notna().sum(axis=1) >= np.ceil(min_items * len(its))
        out[s] = (m.mean(axis=1) * 100).where(ok)
    return out


def cronbach_alpha(df):
    df = df.dropna()
    k = df.shape[1]
    if k < 2 or len(df) < 10:
        return np.nan
    return k / (k - 1) * (1 - df.var(ddof=1).sum() / df.sum(axis=1).var(ddof=1))
