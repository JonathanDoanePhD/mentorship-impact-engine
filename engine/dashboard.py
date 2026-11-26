"""Build output/dashboard.html: a self-contained evaluation dashboard (inline SVG, no JS libs).

Design rules: every estimate has an interval; status is shown as text + shape, not color only;
small cells are suppressed upstream; synthetic data is labeled on every view.
"""
import html
import json
from datetime import date

import numpy as np
import pandas as pd

from .instrument import LABELS, PRIMARY, SECONDARY

E = html.escape


# ---------- small SVG helpers ----------
class Scale:
    def __init__(self, d0, d1, r0, r1):
        self.d0, self.d1, self.r0, self.r1 = d0, d1, r0, r1

    def __call__(self, v):
        return self.r0 + (v - self.d0) / (self.d1 - self.d0) * (self.r1 - self.r0)


def nice_ticks(lo, hi, n=5):
    span = hi - lo
    step = 10 ** np.floor(np.log10(span / n))
    for m in (1, 2, 2.5, 5, 10):
        if span / (step * m) <= n:
            step *= m
            break
    start = np.ceil(lo / step) * step
    return [round(t, 6) for t in np.arange(start, hi + step / 2, step) if t <= hi + 1e-9]


def status(lo, hi):
    if lo > 0:
        return ("up", "Clear gain")
    if hi < 0:
        return ("down", "Clear decline")
    return ("flat", "Uncertain")


# ---------- charts ----------
def forest(imp, wave=12, width=1000):
    rows = imp[imp.wave == wave].copy()
    rows["order"] = rows.outcome.map({o: i for i, o in enumerate(PRIMARY + SECONDARY)})
    rows = rows.sort_values("order")
    row_h, top, left, right = 46, 28, 190, 150
    h = top + row_h * len(rows) + 34
    lo = min(-0.2, rows.g_low.min() - 0.05); hi = max(0.5, rows.g_high.max() + 0.05)
    x = Scale(lo, hi, left, width - right)
    out = [f'<svg viewBox="0 0 {width} {h}" class="chart" role="img" aria-label="Program effects at {wave} months with 95% confidence intervals">']
    for t in nice_ticks(lo, hi, 6):
        out.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{top - 8}" y2="{h - 30}" class="grid"/>'
                   f'<text x="{x(t):.1f}" y="{h - 12}" class="tick" text-anchor="middle">{t:+.1f}</text>'.replace("+0.0", "0"))
    out.append(f'<line x1="{x(0):.1f}" x2="{x(0):.1f}" y1="{top - 8}" y2="{h - 30}" class="zero"/>')
    for i, (_, r) in enumerate(rows.iterrows()):
        y = top + i * row_h + row_h / 2
        if i > 0 and r.role == "secondary" and rows.iloc[i - 1].role == "primary":
            out.append(f'<line x1="8" x2="{width - 8}" y1="{y - row_h / 2:.1f}" y2="{y - row_h / 2:.1f}" class="sep"/>')
        st, txt = status(r.g_low, r.g_high)
        out.append(f'<text x="8" y="{y - 4:.1f}" class="lab">{E(r.label)}</text>'
                   f'<text x="8" y="{y + 12:.1f}" class="sub">{r.role.title()} · n {r.n_program}/{r.n_comparison}</text>')
        out.append(f'<g><title>{E(r.label)}: g = {r.g:.2f} (95% CI {r.g_low:.2f} to {r.g_high:.2f}); '
                   f'complete-case g = {r.g_complete_case:.2f}; smallest detectable effect ≈ {r.mde_g:.2f}</title>'
                   f'<rect x="{x(-r.mde_g):.1f}" y="{y - 11:.1f}" width="{x(r.mde_g) - x(-r.mde_g):.1f}" height="22" class="mde"/>'
                   f'<line x1="{x(r.g_low):.1f}" x2="{x(r.g_high):.1f}" y1="{y:.1f}" y2="{y:.1f}" class="ci prog"/>'
                   f'<circle cx="{x(r.g_complete_case):.1f}" cy="{y:.1f}" r="5" class="cc"/>'
                   f'<circle cx="{x(r.g):.1f}" cy="{y:.1f}" r="6" class="pt prog"/></g>')
        out.append(f'<text x="{width - right + 14}" y="{y + 4:.1f}" class="val">{r.g:+.2f}</text>'
                   f'<text x="{width - 8}" y="{y + 4:.1f}" class="pill {st}" text-anchor="end">{txt}</text>')
    out.append(f'<text x="{x(0):.1f}" y="{top - 14}" class="tick" text-anchor="middle">no difference</text></svg>')
    return "".join(out)


def trajectory(traj, outcome, width=300, height=180):
    t = traj[traj.outcome == outcome]
    left, right, top, bottom = 34, 10, 14, 26
    lo, hi = t.low.min() - 2, t.high.max() + 2
    x = Scale(0, 12, left, width - right); y = Scale(lo, hi, height - bottom, top)
    out = [f'<svg viewBox="0 0 {width} {height}" class="chart" role="img" aria-label="{E(LABELS[outcome])} over time by group">']
    for v in nice_ticks(lo, hi, 4):
        out.append(f'<line x1="{left}" x2="{width - right}" y1="{y(v):.1f}" y2="{y(v):.1f}" class="grid"/>'
                   f'<text x="{left - 5}" y="{y(v) + 4:.1f}" class="tick" text-anchor="end">{v:.0f}</text>')
    for w in (0, 3, 6, 12):
        out.append(f'<text x="{x(w):.1f}" y="{height - 8}" class="tick" text-anchor="middle">{w}{" mo" if w == 12 else ""}</text>')
    for arm, cls, dx in (("comparison", "comp", 1.5), ("program", "prog", -1.5)):
        s = t[t.arm == arm].sort_values("wave")
        if s.empty:
            continue
        pts = " ".join(f"{x(w) + dx:.1f},{y(m):.1f}" for w, m in zip(s.wave, s["mean"]))
        out.append(f'<polyline points="{pts}" class="line {cls}"/>')
        for _, r in s.iterrows():
            out.append(f'<g><title>{arm.title()}, {r.wave} months: {r["mean"]:.1f} (95% CI {r.low:.1f}-{r.high:.1f}), n={r.n}</title>'
                       f'<line x1="{x(r.wave) + dx:.1f}" x2="{x(r.wave) + dx:.1f}" y1="{y(r.low):.1f}" y2="{y(r.high):.1f}" class="ci {cls}"/>'
                       f'<circle cx="{x(r.wave) + dx:.1f}" cy="{y(r["mean"]):.1f}" r="4" class="pt {cls}"/></g>')
    out.append("</svg>")
    return "".join(out)


def interval_rows(df, label_col, mean_col, lo_col, hi_col, fmt="{:+.1f}", width=640, ref=0, cls_fn=None, unit=""):
    row_h, left, right, top = 34, 250, 70, 10
    h = top + row_h * len(df) + 26
    lo = min(ref, df[lo_col].min()); hi = max(ref, df[hi_col].max())
    pad = (hi - lo) * 0.08
    x = Scale(lo - pad, hi + pad, left, width - right)
    out = [f'<svg viewBox="0 0 {width} {h}" class="chart" role="img">']
    for t in nice_ticks(lo - pad, hi + pad, 5):
        out.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{top}" y2="{h - 22}" class="grid"/>'
                   f'<text x="{x(t):.1f}" y="{h - 6}" class="tick" text-anchor="middle">{t:g}{unit}</text>')
    out.append(f'<line x1="{x(ref):.1f}" x2="{x(ref):.1f}" y1="{top}" y2="{h - 22}" class="zero"/>')
    for i, (_, r) in enumerate(df.iterrows()):
        y = top + i * row_h + row_h / 2
        cls = cls_fn(r) if cls_fn else "prog"
        out.append(f'<text x="8" y="{y + 4:.1f}" class="lab">{E(str(r[label_col]))}</text>'
                   f'<g><title>{E(str(r[label_col]))}: {fmt.format(r[mean_col])} (95% CI {fmt.format(r[lo_col])} to {fmt.format(r[hi_col])})</title>'
                   f'<line x1="{x(r[lo_col]):.1f}" x2="{x(r[hi_col]):.1f}" y1="{y:.1f}" y2="{y:.1f}" class="ci {cls}"/>'
                   f'<circle cx="{x(r[mean_col]):.1f}" cy="{y:.1f}" r="5.5" class="pt {cls}"/></g>'
                   f'<text x="{width - 8}" y="{y + 4:.1f}" class="val" text-anchor="end">{fmt.format(r[mean_col])}</text>')
    out.append("</svg>")
    return "".join(out)


def benchmark_chart(ib, width=1000):
    items = ib.item_id.unique().tolist()
    names = ib.drop_duplicates("item_id").set_index("item_id").measure.to_dict()
    row_h, left, right, top = 50, 250, 20, 10
    h = top + row_h * len(items) + 28
    x = Scale(0, 1, left, width - right)
    out = [f'<svg viewBox="0 0 {width} {h}" class="chart" role="img" aria-label="Participants at baseline compared with national benchmarks">']
    for t in (0, .25, .5, .75, 1):
        out.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{top}" y2="{h - 22}" class="grid"/>'
                   f'<text x="{x(t):.1f}" y="{h - 6}" class="tick" text-anchor="middle">{t:.0%}</text>')
    for i, it in enumerate(items):
        y = top + i * row_h + row_h / 2
        s = ib[ib.item_id == it]
        p = s.iloc[0]
        out.append(f'<text x="8" y="{y - 3:.1f}" class="lab">{E(it)} · {E(names[it].split(" (")[0])}</text>'
                   f'<text x="8" y="{y + 12:.1f}" class="sub">{E(p.benchmark_source)}</text>')
        for _, b in s.iterrows():
            g = b.benchmark_group
            cls = "natlow" if ("< 200%" in g or "not reliably" in g) else ("natall" if g.startswith("All") else None)
            if cls is None:
                continue
            out.append(f'<g><title>National, {E(g)}: {b.benchmark_share:.0%} ({b.benchmark_low:.0%}-{b.benchmark_high:.0%})</title>'
                       f'<rect x="{x(b.benchmark_share) - 4:.1f}" y="{y - 4:.1f}" width="8" height="8" class="pt {cls}" transform="rotate(45 {x(b.benchmark_share):.1f} {y:.1f})"/></g>')
        out.append(f'<g><title>Participants at baseline: {p.participants_share:.0%} (95% CI {p.low:.0%}-{p.high:.0%}), n={p.n}</title>'
                   f'<line x1="{x(p.low):.1f}" x2="{x(p.high):.1f}" y1="{y:.1f}" y2="{y:.1f}" class="ci prog"/>'
                   f'<circle cx="{x(p.participants_share):.1f}" cy="{y:.1f}" r="5.5" class="pt prog"/></g>')
    out.append("</svg>")
    return "".join(out)


def power_chart(grid, current_n, width=640, height=260):
    left, right, top, bottom = 46, 120, 14, 36
    x = Scale(grid.n_per_arm_enrolled.min(), grid.n_per_arm_enrolled.max(), left, width - right)
    y = Scale(0, 1, height - bottom, top)
    out = [f'<svg viewBox="0 0 {width} {height}" class="chart" role="img" aria-label="Statistical power by youth per group">']
    for v in (0, .2, .4, .6, .8, 1):
        out.append(f'<line x1="{left}" x2="{width - right}" y1="{y(v):.1f}" y2="{y(v):.1f}" class="{"zero" if v == .8 else "grid"}"/>'
                   f'<text x="{left - 6}" y="{y(v) + 4:.1f}" class="tick" text-anchor="end">{v:.0%}</text>')
    for n in grid.n_per_arm_enrolled.unique():
        out.append(f'<text x="{x(n):.1f}" y="{height - 18}" class="tick" text-anchor="middle">{n}</text>')
    out.append(f'<text x="{(left + width - right) / 2:.1f}" y="{height - 2}" class="tick" text-anchor="middle">Youth enrolled per group</text>')
    out.append(f'<line x1="{x(current_n):.1f}" x2="{x(current_n):.1f}" y1="{top}" y2="{height - bottom}" class="sep"/>'
               f'<text x="{x(current_n) + 4:.1f}" y="{height - bottom - 6}" class="tick">this program</text>')
    labels = []
    for e, cls in ((0.1, "e1"), (0.2, "e2"), (0.3, "e3")):
        s = grid[grid.effect_sd == e].sort_values("n_per_arm_enrolled")
        pts = " ".join(f"{x(n):.1f},{y(p):.1f}" for n, p in zip(s.n_per_arm_enrolled, s.power))
        out.append(f'<polyline points="{pts}" class="line {cls}"/>')
        for n, p in zip(s.n_per_arm_enrolled, s.power):
            out.append(f'<circle cx="{x(n):.1f}" cy="{y(p):.1f}" r="3.5" class="pt {cls}"><title>Effect {e} SD, {n} per group: power {p:.0%}</title></circle>')
        last = s.iloc[-1]
        labels.append([y(last.power) + 4, x(last.n_per_arm_enrolled) + 8, f"{e:.1f} SD effect"])
    labels.sort()
    for i in range(1, len(labels)):  # keep end labels at least 15 px apart
        labels[i][0] = max(labels[i][0], labels[i - 1][0] + 15)
    for ly, lx, txt in labels:
        out.append(f'<text x="{lx:.1f}" y="{ly:.1f}" class="lab">{txt}</text>')
    out.append("</svg>")
    return "".join(out)


def gas_hist(gas_per, width=640, height=200):
    bins = np.arange(20, 82, 5)
    counts, _ = np.histogram(gas_per.t_score.clip(20, 80), bins=bins)
    left, right, top, bottom = 40, 12, 12, 34
    x = Scale(20, 80, left, width - right); y = Scale(0, counts.max() * 1.1, height - bottom, top)
    out = [f'<svg viewBox="0 0 {width} {height}" class="chart" role="img" aria-label="Distribution of goal attainment T-scores">']
    for v in nice_ticks(0, counts.max() * 1.1, 4):
        out.append(f'<line x1="{left}" x2="{width - right}" y1="{y(v):.1f}" y2="{y(v):.1f}" class="grid"/>'
                   f'<text x="{left - 6}" y="{y(v) + 4:.1f}" class="tick" text-anchor="end">{v:g}</text>')
    for b0, c in zip(bins[:-1], counts):
        cls = "bar-hi" if b0 >= 50 else "bar-lo"
        out.append(f'<rect x="{x(b0) + 1:.1f}" y="{y(c):.1f}" width="{x(b0 + 5) - x(b0) - 2:.1f}" height="{y(0) - y(c):.1f}" rx="2" class="{cls}"><title>T-score {b0}-{b0 + 5}: {c} youth</title></rect>')
    for v in range(20, 81, 10):
        out.append(f'<text x="{x(v):.1f}" y="{height - 16}" class="tick" text-anchor="middle">{v}</text>')
    out.append(f'<line x1="{x(50):.1f}" x2="{x(50):.1f}" y1="{top}" y2="{height - bottom}" class="zero"/>'
               f'<text x="{x(50) + 4:.1f}" y="{top + 10}" class="tick">50 = goals met as expected</text>'
               f'<text x="{(left + width - right) / 2:.1f}" y="{height - 2}" class="tick" text-anchor="middle">Goal Attainment T-score at 12 months</text></svg>')
    return "".join(out)


# ---------- page ----------
CSS = """
:root{--bg:#f6f7f9;--surface:#fff;--fg:#121821;--muted:#59626f;--line:#dde1e7;--grid:#e9ecf0;
--prog:#2a78d6;--comp:#eb6834;--natall:#59626f;--natlow:#1baf7a;--e1:#e87ba4;--e2:#2a78d6;--e3:#6250d6;
--up:#0b7a3e;--down:#b42318;--flat:#59626f;--warnbg:#fff4e5;--warnfg:#7a4100;--mde:#eef2f7;
--f-body:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;--f-data:"IBM Plex Mono",ui-monospace,Menlo,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0f1319;--surface:#171c24;--fg:#e6eaf0;--muted:#9ca6b3;--line:#2a313c;--grid:#222a34;
--prog:#3987e5;--comp:#d95926;--natall:#9ca6b3;--natlow:#199e70;--e1:#d55181;--e2:#3987e5;--e3:#9085e9;--up:#4cc38a;--down:#f97066;--flat:#9ca6b3;--warnbg:#3a2a12;--warnfg:#fbd38d;--mde:#1e2631;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#0f1319;--surface:#171c24;--fg:#e6eaf0;--muted:#9ca6b3;--line:#2a313c;--grid:#222a34;
--prog:#3987e5;--comp:#d95926;--natall:#9ca6b3;--natlow:#199e70;--e1:#d55181;--e2:#3987e5;--e3:#9085e9;--up:#4cc38a;--down:#f97066;--flat:#9ca6b3;--warnbg:#3a2a12;--warnfg:#fbd38d;--mde:#1e2631;color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 var(--f-body)}
.wrap{max-width:1180px;margin:0 auto;padding-block:20px 48px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}
.banner{background:var(--warnbg);color:var(--warnfg);border-radius:10px;padding:10px 14px;font-weight:500;margin-bottom:18px}
h1{font-size:clamp(1.5rem,3vw,2.1rem);margin:0;letter-spacing:-.015em;text-wrap:balance}
h2{font-size:1.15rem;margin:0 0 4px}
.lede{color:var(--muted);max-width:72ch;margin:6px 0 0}
.eyebrow{font:500 .75rem var(--f-data);letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
.tiles{display:grid;gap:12px;margin:12px 0 20px}
.tiles.t4{grid-template-columns:repeat(4,minmax(0,1fr))}.tiles.t3{grid-template-columns:repeat(3,minmax(0,1fr))}
@media (max-width:760px){.tiles.t4{grid-template-columns:repeat(2,minmax(0,1fr))}.tiles.t3{grid-template-columns:minmax(0,1fr)}}
.tile{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:12px 14px;min-width:0}
.tile .k{font-size:.8rem;color:var(--muted)}
.tile .v{font:600 1.5rem var(--f-data);font-variant-numeric:tabular-nums;margin-top:2px}
.tile .d{font-size:.8rem;color:var(--muted)}
.tile.flag{border-color:var(--warnfg)}
section{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:18px;margin-top:16px;min-width:0}
section>p.note{color:var(--muted);font-size:.9rem;margin:0 0 10px;max-width:80ch}
.cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,320px),1fr));gap:16px}
.cols3{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,260px),1fr));gap:12px}
.mini h3{font-size:.92rem;margin:0}
svg.chart{width:100%;height:auto;display:block;font-family:var(--f-body)}
.chart .grid{stroke:var(--grid);stroke-width:1}
.chart .zero{stroke:var(--muted);stroke-width:1;stroke-dasharray:3 3}
.chart .sep{stroke:var(--line);stroke-width:1}
.chart .tick{fill:var(--muted);font-size:11px;font-variant-numeric:tabular-nums}
.chart .lab{fill:var(--fg);font-size:13px}
.chart .sub{fill:var(--muted);font-size:11px}
.chart .val{fill:var(--fg);font:500 12px var(--f-data)}
.chart .pill{font-size:12px;font-weight:600}
.chart .pill.up{fill:var(--up)} .chart .pill.down{fill:var(--down)} .chart .pill.flat{fill:var(--flat)}
.chart .mde{fill:var(--mde)}
.chart .ci{stroke-width:2.5;stroke-linecap:round}
.chart .line{fill:none;stroke-width:2}
.chart .pt{stroke:var(--surface);stroke-width:2}
.chart .cc{fill:none;stroke:var(--muted);stroke-width:1.5}
.ci.prog,.line.prog{stroke:var(--prog)} .pt.prog{fill:var(--prog)}
.ci.comp,.line.comp{stroke:var(--comp)} .pt.comp{fill:var(--comp)}
.pt.natall{fill:var(--natall)} .pt.natlow{fill:var(--natlow)}
.line.e1{stroke:var(--e1)} .pt.e1{fill:var(--e1)} .line.e2{stroke:var(--e2)} .pt.e2{fill:var(--e2)} .line.e3{stroke:var(--e3)} .pt.e3{fill:var(--e3)}
.bar-hi{fill:var(--prog)} .bar-lo{fill:var(--natall);opacity:.55}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:.82rem;color:var(--muted);margin:4px 0 8px}
.legend span{display:inline-flex;align-items:center;gap:6px}
.sw{width:11px;height:11px;border-radius:50%;display:inline-block}
.sw.sq{border-radius:2px;transform:rotate(45deg);width:9px;height:9px}
.sw.ring{background:none!important;border:1.5px solid var(--muted)}
.sw.band{border-radius:2px;width:18px}
.tbl{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:.88rem}
th,td{text-align:left;padding:7px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--muted);font-weight:500;font-size:.8rem}
td.num{font-family:var(--f-data);font-variant-numeric:tabular-nums;text-align:right;white-space:nowrap}
.tag{display:inline-block;font-size:.75rem;font-weight:600;border-radius:999px;padding:1px 8px;border:1px solid currentColor}
.tag.up{color:var(--up)} .tag.down{color:var(--down)} .tag.flat{color:var(--flat)} .tag.warn{color:var(--warnfg)}
ul.unc{margin:0;padding-left:20px} ul.unc li{margin:6px 0;max-width:85ch}
footer{color:var(--muted);font-size:.8rem;margin-top:22px;max-width:95ch}
"""


def build_dashboard(root, s, ret, rel, imp, grow, sub, dose, gas_per, gas_dom, traj, ib, grid):
    i12 = imp[imp.wave == 12].set_index("outcome")
    syn = s["synthetic"]
    n_prog, n_comp = s["enrolled"].get("program", 0), s["enrolled"].get("comparison", 0)

    def tile(k, v, d, flag=False):
        return f'<div class="tile{" flag" if flag else ""}"><div class="k">{k}</div><div class="v">{v}</div><div class="d">{d}</div></div>'

    prim_tiles = []
    for o in PRIMARY:
        r = i12.loc[o]
        st, txt = status(r.g_low, r.g_high)
        prim_tiles.append(tile(E(LABELS[o]), f"{r.g:+.2f}", f'<span class="tag {st}">{txt}</span> 95% CI {r.g_low:+.2f} to {r.g_high:+.2f}'))
    status_tiles = "".join([
        tile("Youth enrolled", f"{n_prog + n_comp:,}", f"{n_prog} program · {n_comp} comparison"),
        tile("Still responding at 12 months", f"{s['retention_12m'].get('program', 0):.0%} / {s['retention_12m'].get('comparison', 0):.0%}",
             "program / comparison" + (" · <b>gap flagged</b>" if s["differential_attrition_flag"] else ""), s["differential_attrition_flag"]),
        tile("Matches ending early", f"{s['early_closure_rate']:.0%}", "closed before 6 months"),
        tile("Goals met or exceeded", f"{s['gas_share_at_or_above_expected']:.0%}", f"of youth (mean T-score {s['gas_mean_t']:.0f})"),
    ])
    tiles = (f'<div class="tiles t4">{status_tiles}</div>'
             '<div class="eyebrow" style="margin-top:4px">Primary outcomes · effect at 12 months (Hedges\' g)</div>'
             f'<div class="tiles t3">{"".join(prim_tiles)}</div>')

    # Trajectories
    mini = "".join(f'<div class="mini"><h3>{E(LABELS[o])}</h3>{trajectory(traj, o)}</div>' for o in PRIMARY + SECONDARY)

    # Uncertainty panel, written from the results
    unc = []
    for o in PRIMARY:
        r = i12.loc[o]
        if r.g_low <= 0 <= r.g_high:
            unc.append(f"<b>{E(LABELS[o])}:</b> the estimate ({r.g:+.2f}) is smaller than this sample can reliably detect "
                       f"(about ±{r.mde_g:.2f}). We can't yet tell a small benefit from none.")
    if s["differential_attrition_flag"]:
        unc.append(f"<b>Uneven dropout:</b> 12-month response differs by {s['differential_attrition']:.0%} between groups. "
                   "Estimates are weighted for who stayed, but if those who left differ in ways we didn't measure, the "
                   "results could be biased. Improve follow-up with comparison youth (e.g., incentives, school-based surveying).")
    flagged = imp[imp.sensitivity_flag]
    if len(flagged):
        unc.append("<b>Sensitive to missing-data method:</b> " + ", ".join(E(x) for x in flagged.label.unique()) + ".")
    else:
        unc.append("<b>Missing-data check:</b> weighted and complete-case estimates agree within 0.05 SD for every outcome.")
    if s["low_reliability_scales"]:
        unc.append("<b>Weak scales:</b> " + ", ".join(E(LABELS[x]) for x in s["low_reliability_scales"]) +
                   " hold together poorly (Cronbach's alpha < 0.70). Report their items separately or revise the items.")
    n80 = s["n_for_80pct_power"]
    unc.append(f"<b>Sample size:</b> about {n80.get('0.2')} youth per group are needed for 80% power to detect a 0.2 SD effect "
               f"(the typical size in mentoring research); {('a 0.1 SD effect needs more than 600 per group' if n80.get('0.1') is None else f'{n80.get(chr(48)+chr(46)+chr(49))} for 0.1 SD')}.")
    unc.append("<b>Not answered by these data:</b> long-term outcomes (graduation, college, earnings), whether effects last after the match ends, "
               "and which program components matter most. Relationship-quality and dose comparisons are observational.")
    unc_html = "".join(f"<li>{u}</li>" for u in unc)

    # Subgroups table
    srows = []
    for _, r in sub.iterrows():
        if r.suppressed:
            cell = '<td class="num">—</td><td><span class="tag warn">Suppressed (fewer than 20 per group)</span></td>'
        else:
            st, txt = status(r.g_low, r.g_high)
            cell = f'<td class="num">{r.g:+.2f} ({r.g_low:+.2f} to {r.g_high:+.2f})</td><td><span class="tag {st}">{txt}</span></td>'
        srows.append(f"<tr><td>{E(r.label)}</td><td>{E(r.dimension)}: {E(str(r.group))}</td>{cell}</tr>")

    # Reliability table
    relp = rel.pivot(index="scale", columns="wave", values="alpha")
    rrows = "".join(
        f"<tr><td>{E(LABELS.get(sc, sc))}</td>" + "".join(
            f'<td class="num">{"—" if pd.isna(v) else f"{v:.2f}"}</td>' for v in relp.loc[sc]) +
        f'<td>{"<span class=\'tag warn\'>Below 0.70</span>" if relp.loc[sc].min() < 0.7 else "<span class=\'tag up\'>OK</span>"}</td></tr>'
        for sc in relp.index)

    gas_rows = "".join(f'<tr><td>{E(r.domain)}</td><td class="num">{r.goals}</td><td class="num">{r.met_or_exceeded:.0%}</td></tr>'
                       for _, r in gas_dom.sort_values("goals", ascending=False).iterrows())

    dose_cls = lambda r: "comp" if r.group == "Comparison group" else "prog"
    page = f"""<title>Mentorship Impact Engine</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>{CSS}</style>
<div class="wrap">
{'<div class="banner">Synthetic demonstration data. No real youth are represented. Built to test the pipeline before a program launches.</div>' if syn else ''}
<div class="eyebrow">Mentoring program evaluation · 12-month report · generated {date.today():%B %-d, %Y}</div>
<h1>Mentorship Impact Engine</h1>
<p class="lede">Did youth in the mentoring program gain more than similar youth who weren't yet matched? Effects are shown as standardized differences (Hedges' g) with 95% confidence intervals. Every number comes with how sure we can be about it.</p>
{tiles}

<section><h2>Program effects at 12 months</h2>
<p class="note">Difference between program and comparison youth at 12 months, adjusted for each youth's starting score and site, weighted for who stayed in the study. The shaded band is the smallest effect this sample can reliably detect: estimates inside it can't be told apart from zero.</p>
<div class="legend"><span><i class="sw" style="background:var(--prog)"></i>Weighted estimate</span><span><i class="sw ring"></i>Complete-case check</span><span><i class="sw band" style="background:var(--mde)"></i>Smallest detectable effect</span></div>
<div class="tbl">{forest(imp, 12)}</div></section>

<section><h2>What we can and can't conclude yet</h2><ul class="unc">{unc_html}</ul></section>

<section><h2>Change over time</h2>
<p class="note">Average scale scores (0-100) with 95% intervals at each survey. Primary outcomes first.</p>
<div class="legend"><span><i class="sw" style="background:var(--prog)"></i>Program</span><span><i class="sw" style="background:var(--comp)"></i>Comparison</span></div>
<div class="cols3">{mini}</div></section>

<div class="cols">
<section><h2>Relationships and dose</h2>
<p class="note">12-month change in school belonging (points) by how matches went. Observational: youth whose matches lasted may differ in other ways.</p>
{interval_rows(dose, "group", "mean_change", "low", "high", cls_fn=dose_cls)}</section>
<section><h2>Participant-defined goals</h2>
<p class="note">Each youth set 2-3 goals with their mentor and rated progress from -2 (much less than expected) to +2 (much more). Summary T-scores follow Kiresuk &amp; Sherman.</p>
{gas_hist(gas_per)}
<div class="tbl"><table><thead><tr><th>Goal area</th><th style="text-align:right">Goals</th><th style="text-align:right">Met or exceeded</th></tr></thead><tbody>{gas_rows}</tbody></table></div></section>
</div>

<section><h2>Who the program reaches, compared with national figures</h2>
<p class="note">Share of participants at baseline giving each answer, compared with national surveys using the same question. National figures are context, not a comparison group: different years, populations and, for NSCH, parents answering instead of youth.</p>
<div class="legend"><span><i class="sw" style="background:var(--prog)"></i>Participants (baseline)</span><span><i class="sw sq" style="background:var(--natall)"></i>National, all</span><span><i class="sw sq" style="background:var(--natlow)"></i>National, lower-income or hardship</span></div>
<div class="tbl">{benchmark_chart(ib)}</div></section>

<section><h2>Are gains shared fairly?</h2>
<p class="note">Effects at 12 months within groups. Groups with fewer than 20 youth per arm are suppressed to protect privacy and avoid unstable estimates. Wide intervals are expected; look for patterns, not single numbers.</p>
<div class="tbl"><table><thead><tr><th>Outcome</th><th>Group</th><th style="text-align:right">g (95% CI)</th><th>Status</th></tr></thead><tbody>{''.join(srows)}</tbody></table></div></section>

<div class="cols">
<section><h2>How many youth are needed?</h2>
<p class="note">Simulated power to detect an effect at 12 months, given this program's response rate and how strongly baseline predicts follow-up (r = {s['baseline_r']:.2f}). The dashed line marks 80%.</p>
{power_chart(grid, max(n_prog, 50))}</section>
<section><h2>Do the scales hold together?</h2>
<p class="note">Internal consistency (Cronbach's alpha) for each scale at each survey wave. Below 0.70 means its items may be measuring different things.</p>
<div class="tbl"><table><thead><tr><th>Scale</th>{''.join(f'<th style="text-align:right">{w} mo</th>' for w in relp.columns)}<th>Check</th></tr></thead><tbody>{rrows}</tbody></table></div></section>
</div>

<footer><p><b>Method.</b> Lottery-assigned comparison group (waitlist). ANCOVA of each 12-month scale score on group, baseline score and site; inverse-probability-of-retention weights; robust standard errors; Hedges' g using the baseline standard deviation. Secondary: mixed-effects growth models. Subgroup and dose analyses are exploratory.</p>
<p><b>Privacy.</b> Pseudonymous IDs only; names and goal text are held separately by program staff; groups under 20 suppressed. People, not this dashboard, make decisions about participants and the program.</p>
<p>Benchmarks: National Survey of Children's Health 2024; CDC Youth Risk Behavior Survey 2023. Verify figures against original sources before external use.</p></footer>
</div>"""
    (root / "output" / "dashboard_fragment.html").write_text(page)
    (root / "output" / "dashboard.html").write_text(
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n</head>\n<body>\n'
        + page + "\n</body>\n</html>\n")
