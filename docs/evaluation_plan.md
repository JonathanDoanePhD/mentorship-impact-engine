# Evaluation plan: mentoring program built on bridging adults

> Draft design for a future program. Effect sizes, citations and benchmark figures need
> checking against their original sources before use in funder materials. People, not
> this system, make every decision about participants.

## 1. Logic model

Grounded in Project 02 ("bridging adults"), summarized in [Project 02's conceptual model](../https://github.com/JonathanDoanePhD/mathematics-of-mentorship/blob/main/docs/conceptual_model.md):

| Inputs | Activities | Short-term (0-6 months) | Medium-term (6-18 months) | Long-term |
|---|---|---|---|---|
| Trained mentors from a range of backgrounds and careers; matching; staff support; structured activities | 1:1 or small-group meetings, at least twice a month; introductions to people and opportunities (bridging); goal setting | **Supportive adults** and **bridging ties** (SA items); **relationship quality** (RQ) | **Belonging** (BE), **self-efficacy** (SE), **engagement and persistence** (EN), **goal attainment** (GAS) | Graduation, postsecondary enrollment, early-career earnings (linked records, if partners agree) |

**Why these outcomes:**
- Project 02 found that supportive adults and cross-class ties go most consistently with *school-related* outcomes (engagement, grades, graduation), at least as strongly for disadvantaged youth. Those are the primary outcomes.
- *Emotional* outcomes (hope, hopelessness) went with supportive adults *less* under poverty or hardship, so hope is secondary. The program should not promise wellbeing gains from mentoring alone.
- Access gaps (mentors, structured activities, belonging) were large, so the program measures **access itself** (SA, bridging ties) as an outcome, not just a precondition.

## 2. Research questions

1. **Reach:** who enrolls, and does enrollment reach the youth with the fewest supportive adults (SA1 = no at baseline)?
2. **Relationships:** how long do matches last, how good are they (RQ), and what predicts early closure?
3. **Impact (primary):** compared with a comparison group, do participants gain more in belonging, self-efficacy and engagement by 6 and 12 months?
4. **Bridging:** do participants gain bridging ties (SA3-SA5) and supportive adults (SA1-SA2)?
5. **Goals:** do participants reach the goals they set (Goal Attainment Scaling)?
6. **Equity:** are gains similar across groups (household hardship, gender, race/ethnicity), reported only in aggregate with small groups suppressed?
7. **Dose:** do longer, higher-quality matches go with larger gains? (Observational; interpret with care.)

## 3. Design: comparison groups, strongest first

| Option | When feasible | What it supports | Risks |
|---|---|---|---|
| **A. Lottery for oversubscribed slots, with a waitlist** | More eligible applicants than mentors (common) | Causal estimate of the program's effect | Waitlist youth get served later; needs a fair and transparent lottery that people sign off on |
| **B. Matched comparison** (similar non-participants from partner schools, matched on baseline survey and school records) | Partner schools survey a wider group | Credible if baseline measures are rich | Unmeasured differences in motivation; needs data-sharing agreements |
| **C. National benchmarks** (same items as NSCH, YRBS, PISA) | Always | *Descriptive* context ("participants vs. similar youth nationally") | Not a comparison group: different populations, years and respondents (NSCH is parent-reported) |

The engine supports all three. A lottery, if used, decides *order of service*, never eligibility.
Eligibility criteria are drafted by staff and approved by people. This system doesn't decide
who is eligible or accepted.

**Timing:** baseline (before matching), 3 months (relationship check), 6 months, 12 months.
Comparison youth are surveyed on the same schedule.

**Sample size:** see `output/power/` (simulation-based). With effects near **0.20 SD**, the
average reported in mentoring meta-analyses (Raposa et al. 2019: g = 0.21 across 70 studies;
DuBois et al. 2011: about 0.18-0.21 per later reviews; confirm in the originals),
about **400 youth per arm** gives about 80% power for one primary outcome at 12 months under
ANCOVA. Smaller programs should pool cohorts across years, or rely on per-site description
and goal attainment rather than impact estimates.

## 4. Analysis (implemented in `engine/`)

- **Scoring:** scale means (reverse-coded items flipped), internal consistency (Cronbach's
  alpha) per scale and wave, and an item-level check.
- **Primary impact:** ANCOVA. The 12-month score is regressed on group and baseline score,
  with robust SEs, reported as standardized effects (Hedges' g) with 95% CIs. A mixed-effects
  growth model over all waves is the secondary analysis.
- **Missing data:** attrition is reported by group and baseline characteristics. Primary
  estimates use inverse-probability-of-retention weights; complete-case estimates are shown
  alongside. If the two differ meaningfully, the dashboard flags it.
- **Multiple outcomes:** three primary outcomes are pre-specified (belonging, self-efficacy,
  engagement). Others are labeled exploratory.
- **Goal Attainment Scaling:** each goal is rated -2 to +2 against levels set at baseline;
  summary T-scores follow Kiresuk & Sherman (1968), with 50 meaning "as expected".
- **Equity:** subgroup estimates are shown only where each cell has at least 10 youth; others
  are suppressed.
- **Uncertainty first:** every estimate is shown with its interval. The dashboard states the
  smallest effect the current sample could detect, and lists what the data can't answer yet.

## 5. Data governance and privacy

- **Pseudonymous IDs only** in analysis files (`P0001`...). The key linking IDs to names is
  kept by program staff in a separate, access-controlled system, **never in this repo**.
- **Restricted fields** (goal text written by youth, open-ended answers, sensitive items HO2
  and CX1, demographics) are stored separately from responses. Analysis uses coded versions
  only (goal domain and GAS score; demographic categories).
- **Consent:** parent/guardian consent and youth assent, in plain language, covering what is
  collected, who sees it, how long it is kept and how to withdraw.
- **Safeguarding:** HO2 (persistent sadness) and CX1 (basic needs) are optional. If they are
  used, a staff protocol must say who reviews responses, how quickly, and what support or
  referral follows. Otherwise leave them out.
- **Reporting:** aggregate only, with cells under 10 suppressed. No individual-level data
  goes in dashboards, funder reports or public materials.
- **Retention:** keep identifiable data only as long as the evaluation needs it; then delete
  the ID key, leaving de-identified data.
- This repo holds **only synthetic data** produced by `engine/simulate.py`.

## 6. Decision rules for program staff

Results inform people's decisions; they don't make them.
- **Early closure:** if 3-month relationship quality (RQ mean) is below 3.0, or two
  consecutive meetings are missed, staff check in with that match. This flags; it doesn't
  decide.
- **Continue, adjust or stop components:** reviewed by staff and, where relevant, youth and
  families, using the full dashboard, including the uncertainty panel.
