# Weather file schema 3: one file per plot

Schema 2 sends every phone its whole cell: every plot's id, P30 record and WeatherKit area (about 11 km), plus daily
rain, irradiance and climate fields no rule on the phone reads. Schema 3 sends each phone one small file with only
what its own L5 and L8 rules need.

| | |
|---|---|
| Path | `plots/<plot_key>/latest.json`, next to schema 2 in the job's `--out` folder |
| `plot_key` | First 32 hex characters (128 bits) of HMAC-SHA256(`PLOT_KEY_SECRET`, plot id): `nasa_power_job.plot_key` |
| Written by | `nasa_power_job.py`, daily, only when `PLOT_KEY_SECRET` is set |
| Schema 2 | Still written while `WRITE_SCHEMA2 = True`, until the Swift app reads schema 3 |
| Contract | `schema3.schema.json` (JSON Schema 2020-12); examples in `tests/fixtures/schema3/` |

## Fields and their readers

| Field | Example | Produced by | Read on the phone by |
|---|---|---|---|
| `schema` | `3` | the job | the decoder: which format this is |
| `generated_at` | `"2026-10-14T06:15:00Z"` | the job | sync: how old the file is |
| `l5.water_stress` | `false`, `true`, `null` | `krops_rules.choose_p30` | L6 lowers its confidence when `true`; `null` = unknown, never false |
| `l5.p30_quality` | `"ok"`, `"fallback"`, `"none"` | `choose_p30` | L6: how far to trust the flag; L9 log |
| `l5.p30_mm` | `110.8` or `null` | `choose_p30` | L9 log (the value behind the flag) |
| `l5.p30_start_date`, `l5.p30_end_date` | `"2026-09-14"` | `choose_p30` | L9 log; the age of the P30 when a scan uses it (see below) |
| `l5.p30_source` | `"weatherkit-history"`, `"nasa-power"`, `null` | `choose_p30` | L9 log; which attribution applies |
| `l5.p30_missing_days` | `{"weatherkit-history": 1, "nasa-power": 0}` | `choose_p30` (`missing_in_p30_window`) | L9 log: why a fallback happened |
| `stage_a.<1–12>.avg_mm` | `{"10": {"avg_mm": 210.0}}` | the job (`month_table`, NASA POWER 1991–2020) | `krops_rules.l8_decision`: `month_gate` on this month, and on later months while `CUT_WINDOW_AT_CLOSED_MONTH` is on |
| `attribution[]` | `{"source", "text", "url"}` | the job | shown with the data: Apple Weather when `l5` came from WeatherKit; NASA POWER always (Stage A) |

L8 Stage B's forecast is not in the file: the phone reads WeatherKit's forecast itself.

## Left out (schema 2 field → why)

| Schema 2 field | Why it isn't in schema 3 |
|---|---|
| `cell` (id, lat, lon, plots) | Location and other plots' ids; no rule reads it |
| other plots' `l5` records | Not this phone's business |
| `l5.weatherkit_area` | A location (0.1° box); no rule reads it |
| `weatherkit_rain`, `recent_rain` | The backend already turned them into `l5` |
| `irradiance` | Logged only; no rule reads it |
| `climate.normals_mm` | Used by the backend's monthly check only |
| `climate.stage_a.*.gate` | `l8_decision` works the gate out from `avg_mm` |
| `climate.stage_a.*.in_band_share` | Logged only, by decision |
| `climate.period`, `climate.computed_on`, `nasa_power_sources` | Provenance; `attribution` carries what the app must show |

## Serving

The repo doesn't say how the `--out` folder is served. If it is a static folder:

- Serve only `plots/`.
- Keep `plots.json`, `cells/`, `archive/` and every `qa_log.json` off the public path. `plots.json` lists every plot
  with its cell and 0.1° area, cell ids follow a regular grid, and archived forecasts carry the plot's exact
  coordinates under its id.
- Turn off directory listings.

A `plot_key` is a bearer name: whoever has it can read that plot's P30 and month table, nothing else. Changing
`PLOT_KEY_SECRET` gives every plot a new folder, so the app would need its new key.

## Open for the Swift side

- How the app learns its `plot_key`: the backend can compute it from the plot id at enrolment; the secret never
  leaves the server.
- A P30 record ages while a phone is offline. Today no on-device rule checks `p30_end_date` against the scan date
  (the backend applies the 7-day limit when it writes the file). Re-checking it on the phone would be a new rule
  and needs a team decision.
- Decode against `schema3.schema.json` and the three fixtures: `normal`, `weatherkit_fallback`, `null_p30`.
