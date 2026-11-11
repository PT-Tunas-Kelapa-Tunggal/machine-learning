# fertilization_timing: the year's fertilizer rounds for the "Jadwal Pemupukan" screen

One request in (the plot's coordinates, how often it fertilizes, which fertilizers and how much a year), one JSON out: the rounds of
the plan year, each with its two-month window, the products in order with the wait between them, and, for the round
under way or about to start, the day to spread each product or the reason to wait.

| | What it is |
|---|---|
| `fertilization_timing.py` | the schedule: pure functions and a command line |
| `example_request.json` | an input: Dolomite, five NPK grades and Urea, 2 rounds a year, the Tanjung Pinang trial site; five products have no `id` |
| `example_output.json` | its output on 8 Nov 2026 (no round active, so no daily dates) |
| `example_request_in_progress.json` | Dolomit + NPK + Urea (400 / 120 / 100 kg a round), Dolomit marked 5 Jan 2027, NPK 19 Jan |
| `example_output_in_progress.json` | its output on 19 Jan 2027 with `--no-daily`: the card of image 8 (Urea "Bisa ditandai mulai 20 Januari") |
| `test_fertilization_timing.py` | unit tests with a fixed climate table and made-up weather: no network |
| `krops_rules.py` | the team's L5/L8 weather rules, a verbatim copy (see Copied from) |
| `weather_sources.py` | NASA POWER climate and P30, WeatherKit forecast: the team's code, copied (see Copied from) |
| `requirements.txt`, `.env.example` | what the weather needs: packages, WeatherKit keys |

The folder runs on its own: it imports nothing from the rest of the repo.

## Run

Python 3.9 or later. The schedule and the tests need only the standard library; the weather needs
`requirements.txt`. For the daily weather, copy `.env.example` to `.env` and fill in the WeatherKit keys (or point
`KROPS_WEATHER_ENV` to a `.env` elsewhere). Without them there is no forecast, which the flags say.
The climate is cached in `data/weather_cache/`. Only the trial site's cell (`n1.000_e104.3750`, computed 26 Oct
2026) is committed, so the examples run offline until it turns a year old (26 Oct 2027); any other plot's first run
fetches 30 years of NASA POWER rain for its cell, once a year.

```bash
pip install -r requirements.txt
python fertilization_timing.py --request example_request.json \
    [--today 2026-11-08] [--year 2027] [--no-daily] [--out schedule.json]
python -m unittest discover -s .
```

- `--today`: the day to plan from (default: today in WIB). It sets the plan year, the status of each round and which
  round is active.
- `--year`: force the plan year. Default: this year while one of its rounds hasn't ended, else next year.
- `--no-daily`: no forecast or P30 calls; every `daily` stays `null`.
- `--cache-dir`: where the climate is cached (default `data/weather_cache/` here).

## Copied from

The weather code is the team's, copied on 10 Nov 2026 (commit `f758e90`) so the folder runs on its own. Only the
module prefixes (`job.`, `l2.`, `l6.`) were dropped; the code is otherwise unchanged.

| Here | From |
|---|---|
| `krops_rules.py` | `weather/krops_rules.py`, whole and byte for byte. `CopiedRulesTest` fails when the two differ and is skipped outside the repo |
| `weather_sources.py`: the grid cells, `fetch_nasa_power`, `WeatherKit` (without `hourly`), `build_climate`, `build_recent`, `load_climate`, `write_json` | `weather/nasa_power_job.py` |
| `weather_sources.py`: `p30_at`, `scan_p30` | `weather/l2_scan_rain.py`, `weather/l6_vari_confidence.py` |
| `weather_sources.py`: `WeatherClient` (without `flight_confidence`), `load_env` (reads `.env` here) | `integration/krops_vari/weather.py` |
| `fertilization_timing.py`: `wait_reason`, `KIESERITE_WAIT_DAYS` | `integration/krops_vari/spreading.py`, `config.py` (`min_days_mg_to_n`) |

`krops_rules.py` keeps `CUT_WINDOW_AT_CLOSED_MONTH = True` as upstream: the schedule switches the cut off where it
calls `l8_decision` (Rules and their status: Safe day).

When one of these changes upstream, copy it over again: `cp weather/krops_rules.py integration/v2/fertilization_timing/`,
and the matching functions by hand for the rest.

## Datapoints

### From the app (the request JSON)

| Field | Required | Used for |
|---|---|---|
| `latitude`, `longitude` | **yes** | the climate cell and the forecast; the run stops when either is `null`. Never written to the output |
| `fertilizerFreq` | **yes** | rounds a year: 1, 2, 3, 4 or 6 (12 ÷ freq whole months apart) |
| `fertilizers[].name` | **yes** | the product: sets its order in a round, its wait and its display name |
| `fertilizers[].overallDoseWeightInKg` | **yes** | kg of that product for the whole farm **in a year**; each round gets it ÷ freq |
| `fertilizers[].id` | yes, with `applications` | matches a spread step to its product. `null` is allowed: the step is still planned but can't be marked, is left out of `completed`, and is flagged |
| `fertilizers[].composition` | NPK only | the grade in the name: "NPK 15-15-15" (Mg added when not 0) |
| `fertilizers[].bagKg` | no | passed through as `bag_kg` |
| `applications[]`: `{fertilizerId, date}` | no | the steps already spread ("Tandai Mulai Memupuk", and "Tandai Sudah Dipupuk" for an overdue one). A date belongs to the round it falls in or after, up to the next round's start, so a step marked after its window still counts. The card states and the next step's dates count from it. A date in the plan year before its first round is ignored and flagged |

Products with a rule: Dolomite/Dolomit, Kieserite/Kieserit, NPK, SP-36, KCl/MOP, Urea, ZA. Any other name is kept,
placed after NPK and before Urea/ZA, and flagged.

### Weather

| Data | Source | Used for | Calls | Cache |
|---|---|---|---|---|
| Monthly rain 1991–2020: average and in-band (100–200 mm) share per calendar month | NASA POWER daily `PRECTOTCORR`, grid cell 0.5° × 0.625° (free, no key) | L8 Stage A: which months are open, which months the rounds go in. Also the P30 check below | 1 per cell a year | `data/weather_cache/cells/<cell>/climate.json`, 365 days |
| 10-day daily rain forecast | Apple WeatherKit `forecastDaily` (keys in `.env`) | L8 Stage B: the safe days of the active round | 1 per run with an active round | in memory only |
| P30: rain of the last 30 days | WeatherKit daily history (≈11 km area) first, NASA POWER (checked: implausibly wet months dropped) as the fallback | L5: under 60 mm → wait for rain | WeatherKit 5–8 + NASA POWER 1 per run with an active round | none |

Weather is only fetched when a round is active: under way, or starting within 10 days. On 8 Nov 2026 with the
example request (rounds in March and September 2027) the run makes no weather call at all. A weather failure
never stops the run: it becomes a flag naming the error type only (a request URL holds the coordinates). Without
WeatherKit there is no forecast, so L8 has no safe day and every step waits: the flags say so.

### Rules and their status

| Rule | Value | Status | Where |
|---|---|---|---|
| Open month | average rain 100–300 mm; 60–100 mm `cek`; under 60 or over 300 `tunda` | proposal / decided 9 Oct 2026 | `krops_rules.py` `month_gate` |
| Round months | open months 12 ÷ freq apart, highest in-band share; a tie goes to the earliest start | from `ml-integration` 154ac1f; the tie rule is this script's | `round_months` |
| Round window | its month and the next one (Jan–Feb = 59 days) | decided 11 Nov 2026 (from the app design) | `ROUND_WINDOW_MONTHS` |
| Safe day | forecast day ≤ 25 mm, not the day before a heavy day, not the last forecast day, while today's month is open | proposal / decided 9 Oct 2026; no cut at a closed month: a safe day in the window's next month stays (decided 11 Nov 2026, `cut_at_closed_month=False`) | `krops_rules.l8_decision` |
| Too dry | P30 under 60 mm → wait for rain | decided | `krops_rules.choose_p30`, `wait_reason` |
| Order in a round | Dolomit/Kieserit → NPK → SP-36/KCl → Urea/ZA | practice | `PRODUCTS` |
| After Dolomit | 14–28 days | team decision (11 Nov 2026); the mechanism is peer-reviewed, the days are practice (below) | `DOLOMITE_WAIT_DAYS` |
| After Kieserit | 1 day | decided 11 Nov 2026 (as in `integration/krops_vari/config.py`) | `KIESERITE_WAIT_DAYS` |
| Before Urea/ZA | any later day | no source sets a gap from NPK, KCl or SP-36 | `BEFORE_N_STRAIGHT_WAIT_DAYS` |
| Between other products | 1–7 days | decided 11 Nov 2026 (from the app design) | `OTHER_WAIT_DAYS` |

A daily date uses the minimum wait: the next step goes on the first safe day at least that many days after the
previous step's day.

Sources for the waits (checked 8 Nov 2026):
- Dolomite is a lime: the higher pH near its particles turns urea N into ammonia gas. Each t/ha of dolomitic lime
  raised the NH3 loss from urea by 1.7–2.2% ([Minato et al., RBCS 2026](https://www.rbcsjournal.org/article/liming-and-enhanced-efficiency-nitrogen-fertilizers-ammonia-volatilization-losses-from-subtropical-no-till/)).
  The paper gives no gap in days.
- 21–28 days before urea, ZA or NPK; 14–21 before SP-36; 14 before KCl: a dolomite producer
  ([pabrikpupukdolomit.com](https://pabrikpupukdolomit.com/jarak-pemberian-dolomit-dengan-pupuk-lain-pada-sawit/)).
- "Separate lime and urea applications in time", no number ([Chemiseed](https://chemiseed.com/en-id/blogs/cover-crops-seeds/liming-acid-soils-dolomite-oil-palm-malaysia-ph-calcium-magnesium)).
- A TM 15 trial in Bungo spread urea together with KCl, a week after SP-36 ([Rozi & Prastia 2019](https://ojs.umb-bungo.ac.id/index.php/saingro/article/view/243)).
- No PPKS or MPOB number was found for any of these gaps: an agronomist should confirm them.

## Output

| Field | Meaning |
|---|---|
| `plan_year`, `today`, `rounds_per_year` | the header: "Rekomendasi untuk tahun 2027" |
| `rounds[].start_month_name`, `end_month_name`, `start_date`, `end_date`, `total_days` | the round card: "Mulai Januari · Selesai Februari" |
| `rounds[].status` | `upcoming` (`days_until_start`: "Dimulai dalam 54 hari lagi"), `ongoing` (`day_of_round` of `total_days`: "Hari ke-5 dari 59 hari"), `done` |
| `rounds[].card` | the card header state: see Card states |
| `rounds[].steps[]` | `display_name`, `dose_kg` (this round), `annual_dose_kg`, `bag_kg`, `fertilizer_id`, `card` (see Card states), `wait_after_days` `{min, max, state, until}` before the next step (`max: null` = any later day; `null` on the last step) |
| `rounds[].daily` | `null` unless the round is active and not completed. Else: `l5` (the P30 record), `l8` (`state`, `month_gate`, `safe_days`, `removed`), `wait` (the reason to wait, or `null`), `reference_day` (today, or the round's start when it hasn't started) |
| `rounds[].steps[].schedule` | the weather's suggestion, as small text on the card (the buttons stay enabled); active rounds only: `status` `done` / `planned` (`date`) / `not_before` (past the forecast: run again then) / `wait` (`wait` holds the reason) |
| `weather_basis` | the month table, the candidate month sets and the rule |
| `flags` | what needs a look: unknown products, two Mg sources, a window running into a closed month, no forecast, … |

The daily part of a live run on 8 Nov 2026 (Dolomite 22 kg + NPK 15-15-15 24 kg a year with `fertilizerFreq: 6`, so a November–December round is under way):

```json
"daily": {
 "checked_on": "2026-11-08",
 "reference_day": "2026-11-08",
 "l5": {"p30_mm": 143.0, "p30_start_date": "2026-10-09", "p30_end_date": "2026-11-07",
        "p30_source": "weatherkit-history", "water_stress": false, "p30_quality": "ok", "...": "..."},
 "l8": {"state": "jendela_tabur", "month_gate": {"month": 11, "avg_mm": 298.8, "gate": "open"},
        "safe_days": ["2026-11-08", "2026-11-09", "...", "2026-11-16"],
        "removed": {"heavy": [], "day_before_heavy": [], "last_cached": ["2026-11-17"]}},
 "wait": null
},
"steps": [
 {"display_name": "Dolomit", "dose_kg": 3.667, "...": "...",
  "schedule": {"status": "planned", "date": "2026-11-08", "not_before": null, "wait": null}},
 {"display_name": "NPK 15-15-15", "dose_kg": 4.0, "...": "...",
  "schedule": {"status": "not_before", "date": null, "not_before": "2026-11-22",
               "wait": "no safe day that far ahead in the forecast: run again then"}}
]
```

14 days after Dolomit is past the 10-day forecast, so NPK gets a "not before" date: the run on or after
22 November gives its day. With `applications` saying Dolomit went on 8 November, that run counts from the real date.

## Card states

Every round carries the states of its card, worked out from the dates alone (no weather, no network), so the app
renders them as they are. Image numbers are the design's (Dolomit 400 kg → NPK 15-15-15 120 kg → Urea 100 kg).

**Header** (`rounds[].card`)

| `state` | When | Data | Screen |
|---|---|---|---|
| `upcoming` | before the round starts | `days_until_start` | "Dimulai dalam 54 hari lagi" (image 4) |
| `ongoing` | in the round, nothing marked yet | `day_of_round`, `total_days` | "Berlangsung · Hari ke-5 dari 59 hari" (5) |
| `started` | something marked, not everything, the round not over | `started_on`: the first date marked | "Dimulai 5 Jan 2027" (6–9) |
| `period_passed` | the round is over, something isn't marked | `period_end` | "Periode terlewat 28 Feb 2027" (10) |
| `completed` | every step with an `id` marked (even after the round) | `completed_on`: the last date marked | the collapsed card "Selesai 22 Januari 2027" |

**Each fertilizer** (`rounds[].steps[].card`)

| `state` | When | Data | Screen |
|---|---|---|---|
| `locked` | the round hasn't started, or the step before isn't marked | – | name and kg only (4, 5) |
| `ready` | it can be marked and its deadline hasn't passed | `available_from`, `deadline` | "Tandai Mulai Memupuk" (5, 7, 9) |
| `waiting` | the step before is marked, its minimum wait isn't over | `available_from` | "Bisa ditandai mulai 19 Januari" (6, 8) |
| `done` | in `applications` | `done_on` | "Dimulai 5 Januari" (6–10) |
| `overdue` | today is past its deadline | `overdue_since`: the deadline | "Terlewat sejak 26 Januari" + "Tandai Sudah Dipupuk" (10) |

A step can be marked from the step before's date + the minimum wait (the first step: the round's start). Its
deadline is that date + the maximum wait, or the round's end when that comes first, when the wait has no maximum
(before Urea/ZA), or for the first step. So any fertilizer can become overdue: NPK 28 days after Dolomit while the
round still runs, Dolomit or Urea when the round ends.

**The wait between two fertilizers** (`rounds[].steps[].wait_after_days`)

| `state` | When | Data | Screen |
|---|---|---|---|
| `idle` | the step above isn't marked, or the header is `upcoming`, `period_passed` or `completed` | – | "Tunggu 14–28 hari" |
| `counting` | the step above is marked, the minimum wait isn't over | `until`: the next step's `available_from` | "· sampai 19 Januari" or "· mulai 20 Januari" |
| `done` | the minimum wait is over | – | "· selesai" |

The words are the app's: the design says "sampai" after a fixed wait and "mulai" after a range, both for the same
date. The weather (`schedule`, active rounds only) is a suggestion shown as small text: the buttons follow the dates
above, never the weather.

## Not covered

- The months come from 1991–2020 averages: an unusually wet or dry year doesn't move them, only the daily dates.
- The doses are the farmer's own (`overallDoseWeightInKg`), not the VARI boost or the NPK baseline gap from the
  iteration-2 notebook.
- A round starting in December (window December–January) is planned in the year it starts only.
