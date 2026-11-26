# Data schema

The pipeline reads five CSV files. A real program exports its data in this layout to a
folder **outside this repository** and runs:

```bash
python3 -I scripts/run_pipeline.py --data /secure/path/to/export
```

Identifiers are pseudonymous. The table linking IDs to names, contact details and consent
records lives only in the program's secure system (see `evaluation_plan.md`, section 5).

## participants.csv: one row per youth

| Column | Type | Notes |
|---|---|---|
| `pid` | text | Pseudonymous ID, e.g. `P0001`. Never a name, email or student number. |
| `arm` | `program` / `comparison` | Group from the lottery or comparison design |
| `site` | text | Site or school label (use codes if a site is very small) |
| `cohort` | integer | Enrollment year |

## demographics.csv: restricted, one row per youth

Kept separate from responses and used only for aggregate equity analysis (groups under 20
suppressed). Categories only, no free text.

| Column | Type | Notes |
|---|---|---|
| `pid` | text | |
| `hardship` | 0/1 | From baseline item CX1 (basic needs not reliably met = 1), or a program-defined indicator such as free/reduced-price lunch eligibility |
| `gender` | category | Program's own categories, including a self-described option |
| `grade` | integer | |
| `raceeth` | category | Program's own categories |

## responses.csv: long format, one row per answer

| Column | Type | Notes |
|---|---|---|
| `pid` | text | |
| `wave` | 0 / 3 / 6 / 12 | Months since baseline |
| `item_id` | text | From `instrument/items.csv` |
| `code` | integer | Answer code from `instrument/response_sets.csv` |

Skipped questions are simply absent. Open-ended answers do **not** go in this file.

## matches.csv: one row per mentoring match (program youth)

| Column | Type | Notes |
|---|---|---|
| `match_id` | text | |
| `pid` | text | |
| `mentor_id` | text | Pseudonymous mentor ID |
| `months_active` | number | Months from match start to close (or to now) |
| `closed_early` | 0/1 | Closed before 6 months |
| `meetings` | integer | Logged meetings |

## goals.csv: Goal Attainment Scaling (program youth)

| Column | Type | Notes |
|---|---|---|
| `pid` | text | |
| `goal_id` | text | `G1`-`G3` |
| `domain` | category | School; College & career; Relationships; Wellbeing; Skills |
| `wave` | 6 / 12 | |
| `gas_score` | -2..+2 | Rated against the 5 levels the youth and mentor wrote at baseline |

The goal text and level descriptions youth write are personal. Keep them in the program's
secure system, not in this export.

## Goal Attainment Scaling, briefly

At baseline each youth writes 2-3 goals with their mentor. For each goal they describe five
levels: **-2** much less than expected, **-1** somewhat less, **0** the expected outcome,
**+1** somewhat more, **+2** much more. At 6 and 12 months they rate where they landed. The
summary T-score (Kiresuk & Sherman 1968) is
`T = 50 + 10 * sum(x) / sqrt(0.7 * k + 0.3 * k^2)` for k equally weighted goals: 50 means
goals were met as expected.
