# integration: the VARI pipeline as notebooks and as a backend package

| | What it is |
|---|---|
| `integration.ipynb` | the teammates' VARI notebook, split by stage, with zone confidence: the explanation and the figures |
| `weather_fertilizer.ipynb` | the weather rules (L5, L8) joined to the doses: which fertilizer, which day |
| `krops_vari/` | the same logic as a Python package, for the backend: `run_scan`, a command line and an HTTP API |
| `tests/` | unit tests, and a golden test: `dji_10.jpg` through the package equals the notebook's output |

## Install

macOS only: the palm detector is a Core ML model.

```bash
conda activate vari-confidence
pip install -e "integration[api]"     # editable: changes to krops_vari/*.py apply without reinstalling
```

WeatherKit keys are read from `weather/.env`, as for the weather job.

## The package

```
krops_vari/
├── config.py        Settings: every constant, passed to each stage
├── segmentation.py  1. ExG + Otsu leaf mask
├── detection.py     2. Core ML palm detection (loaded once) and the edge filter
├── vari.py          3. VARI per pixel and per box
├── zoning.py        4–5. trees into zones; zone VARI over the members' leaf pixels
├── confidence.py    zone confidence: tree factor × leaf factor × flight confidence
├── boost.py         P95/P25 from reference zones, boost, Mg/N per zone, product mass
├── weather.py       the weather/ code (climate, P30, forecast, L6) behind one client
├── spreading.py     this round's products and dates, the yearly calendar
├── pipeline.py      run_scan(): every stage in order → a JSON-ready result
├── main.py          command line
└── api.py           HTTP API (FastAPI)
```

The stage modules don't print, plot or read globals: they take inputs and a `Settings`, and return data.
`weather.py` imports the rules from `weather/` (`krops_rules.py`, `l6_vari_confidence.py`, `nasa_power_job.py`)
rather than copying them.

## Command line

```bash
cd integration
# the photo alone (the same result as integration.ipynb)
python -m krops_vari.main --photo ../segmentation/final_flow_ut1/dji_10.jpg --out result.json

# with the plot's weather: the spreading plan; with flight times also the L6 flight confidence
python -m krops_vari.main --photo dji_10.jpg --plot-id P01 --plots plots.json \
    --flight-start 2026-10-26T03:00:00Z --flight-end 2026-10-26T03:20:00Z --out result.json
```

`plots.json` is a list of `{plot_id, lat, lon}` and stays on the server. The result never contains coordinates.

## HTTP API

```bash
uvicorn krops_vari.api:app --host 127.0.0.1 --port 8000
```

The server starts on its own: no environment variables and no plot file are needed. Every setting below is optional.

| Variable | Default | What it is for |
|---|---|---|
| `KROPS_PLOTS` | `weather/plots.json` (a missing file is fine) | plot locations, for the weather features |
| `KROPS_DATA_DIR` | `integration/data/` (gitignored) | uploads, results and the weather cache |
| `KROPS_SETTINGS` | none: the notebook's values | a JSON file overriding `Settings` |
| `KROPS_NO_WEATHERKIT` | unset | `1` = NASA POWER only (no forecast) |

WeatherKit keys are read from `weather/.env`; without them, the weather features use NASA POWER only.

A photo-only scan needs none of these. The plot file only adds the weather features:

| A scan with | Needs the plot file | Adds |
|---|---|---|
| a photo | no | zones, confidence (0.60 stands in for the flight part), boost, Mg/N product mass |
| + `plot_id` | yes | the spreading plan (P30, the month gate, safe days, which product on which day) |
| + `plot_id` and flight times | yes | the L6 flight confidence, which replaces the 0.60 |

The plot file is a JSON list of `{plot_id, lat, lon}`. The weather is looked up by location, but the app sends only
the `plot_id`, so the coordinates stay on the server. Keep the file outside the repo, e.g. `~/krops/plots.json` on a Mac:

```bash
KROPS_PLOTS=~/krops/plots.json KROPS_DATA_DIR=~/krops/data \
    uvicorn krops_vari.api:app --host 127.0.0.1 --port 8000
```

A `plot_id` the file doesn't list (or no file at all) gets `404 unknown plot_id`. The file is read on each request,
so new plots need no restart. Use `--host 0.0.0.0` only on a trusted network: the API has no login yet.

| Request | Answer |
|---|---|
| `POST /scans` (multipart: `photo`; optional `plot_id`, `flight_start`, `flight_end`, `scan_id`) | `202 {"scan_id", "status": "queued"}` |
| `GET /scans/{scan_id}` | `{"status": queued / running / done / failed, "result"}` |
| `GET /health` | `{"status", "version", "model_loaded"}` |

Scans run in the background, one at a time, with the model loaded once at startup. Uploads and results go to
`KROPS_DATA_DIR` (default `integration/data/`, gitignored), so a result survives a restart.

## The result

| Key | Holds |
|---|---|
| `zones` | per zone: trees, pixel box, VARI, `confidence` (factors, reference, flags), `recommendation` (boost, Mg/N) |
| `summary` | reference zones, P25, P95, boost mode |
| `product_mass` | kg of each product per zone and per palm, per year |
| `flight_confidence` | the L6 result when flight times and a plot were given, else `null` |
| `spreading` | L5 P30, L8 state and safe days, this round per zone and product, the yearly calendar; `null` without a plot |
| `run_flags` | what may make the result less accurate, including any weather source that failed |

## Tests

```bash
cd integration && python -m unittest discover -s tests
```

`test_golden.py` and `test_api.py` run the real model on `dji_10.jpg` and compare with
`tests/fixtures/dji_10_notebook.json`, the notebook's own output. They are skipped off macOS.

## Third-party calls per scan with a plot

| API | Calls | For |
|---|---|---|
| WeatherKit | 1 forecast + 5–8 history (+1 hourly with flight times) | L8 Stage B, P30, L6 |
| NASA POWER | 1–2 (+1 per grid cell a year, cached) | P30 check and fallback, the 1991–2020 climate |

With a flight on the same day, P30 comes from the L6 call and isn't fetched twice.
