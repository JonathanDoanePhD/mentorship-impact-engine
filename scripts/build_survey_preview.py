"""Render the item bank as a youth-facing survey preview for piloting (wording, length,
reading level). python3 -I scripts/build_survey_preview.py -> output/survey_preview.html

Sections follow the constructs; program-only and sensitive items are marked for staff.
The preview is not a data-collection form: answers typed into it are not saved anywhere.
"""
import html
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
items = pd.read_csv(ROOT / "instrument" / "items.csv")
rs = pd.read_csv(ROOT / "instrument" / "response_sets.csv")
E = html.escape

SECTION_INTRO = {
    "Supportive adults": "These questions are about adults in your life outside your family.",
    "Bridging ties": "These questions are about people and opportunities you know about.",
    "Mentor relationship": "Only for youth who have a mentor in the program.",
    "School belonging": "How much do you agree with each statement about your school?",
    "Self-efficacy": "How sure are you that you can do each of these things?",
    "Engagement and persistence": "How often is each of these true for you?",
    "Hope": "",
    "Future orientation": "",
    "Context (optional)": "Optional. You can skip this question.",
}

blocks, n = [], 0
for scale, g in items.groupby("scale", sort=False):
    qs = []
    for _, it in g.iterrows():
        n += 1
        opts = rs[rs.response_set == it.response_set]
        radios = "".join(
            f'<label class="opt"><input type="radio" name="{E(it.item_id)}" id="{E(it.item_id)}_{c}"> {E(str(l))}</label>'
            for c, l in zip(opts.code, opts.label))
        tags = []
        if it.who == "program":
            tags.append('<span class="tag">Program youth only</span>')
        if it.sensitive == 1:
            tags.append('<span class="tag warn">Sensitive: needs a staff follow-up protocol</span>')
        if isinstance(it.benchmark, str) and it.benchmark:
            tags.append(f'<span class="tag bench">National benchmark: {E(it.benchmark.split(" (")[0])}</span>')
        qs.append(f'<fieldset><legend><span class="num">{n}.</span> {E(it.text)}</legend>'
                  f'<div class="opts">{radios}</div><div class="meta"><code>{E(it.item_id)}</code> · '
                  f'{E(it.source)} {"".join(tags)}</div></fieldset>')
    intro = SECTION_INTRO.get(scale, "")
    blocks.append(f'<section><h2>{E(scale)}</h2>{f"<p class=intro>{E(intro)}</p>" if intro else ""}{"".join(qs)}</section>')

page = f"""<title>Youth Survey Preview</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:wght@400;700&family=IBM+Plex+Mono&display=swap">
<style>
:root{{--bg:#f6f7f9;--surface:#fff;--fg:#121821;--muted:#59626f;--line:#dde1e7;--accent:#2a78d6;--warnfg:#7a4100;
--f:"Atkinson Hyperlegible",system-ui,-apple-system,"Segoe UI",sans-serif;--mono:"IBM Plex Mono",ui-monospace,Menlo,monospace}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#0f1319;--surface:#171c24;--fg:#e6eaf0;--muted:#9ca6b3;--line:#2a313c;--accent:#3987e5;--warnfg:#fbd38d;color-scheme:dark}}}}
:root[data-theme="dark"]{{--bg:#0f1319;--surface:#171c24;--fg:#e6eaf0;--muted:#9ca6b3;--line:#2a313c;--accent:#3987e5;--warnfg:#fbd38d;color-scheme:dark}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--fg);font:17px/1.55 var(--f)}}
.wrap{{max-width:760px;margin:0 auto;padding-block:24px 48px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}}
h1{{font-size:1.8rem;margin:0}} .lede{{color:var(--muted)}}
.staff{{border:1px dashed var(--line);border-radius:10px;padding:10px 14px;font-size:.9rem;color:var(--muted)}}
section{{margin-top:28px}} h2{{font-size:1.25rem;margin:0 0 4px}} .intro{{color:var(--muted);margin:0 0 10px}}
fieldset{{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:14px 16px;margin:0 0 12px;min-width:0}}
legend{{font-weight:700;padding:0;float:left;width:100%;margin-bottom:10px}} .num{{color:var(--muted);font-weight:400}}
.opts{{clear:both;display:flex;flex-wrap:wrap;gap:8px}}
.opt{{display:inline-flex;align-items:center;gap:6px;border:1px solid var(--line);border-radius:999px;padding:6px 12px;cursor:pointer}}
.opt:has(input:checked){{border-color:var(--accent);outline:1px solid var(--accent)}}
input:focus-visible{{outline:2px solid var(--accent);outline-offset:2px}}
.meta{{margin-top:10px;font-size:.78rem;color:var(--muted)}} code{{font-family:var(--mono)}}
.tag{{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:0 8px;margin-left:6px}}
.tag.warn{{color:var(--warnfg);border-color:currentColor}} .tag.bench{{color:var(--accent);border-color:currentColor}}
</style>
<div class="wrap">
<h1>Youth Survey Preview</h1>
<p class="lede">{n} questions across {items.scale.nunique()} topics. About 8-10 minutes. There are no right or wrong answers, and you can skip any question.</p>
<p class="staff"><b>For staff:</b> preview for piloting wording, length and reading level. Answers typed here are not saved. Grey lines under each question show its ID, source and any national benchmark; remove those lines from the youth-facing version.</p>
{''.join(blocks)}
</div>"""
(ROOT / "output" / "survey_preview.html").write_text(
    '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
    '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n</head>\n<body>\n'
    + page + "\n</body>\n</html>\n")
print(f"{n} items -> output/survey_preview.html")
