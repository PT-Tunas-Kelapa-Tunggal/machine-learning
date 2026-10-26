"""HTTP API: upload a photo, the scan runs in the background, fetch the result by id.

    uvicorn krops_vari.api:app --host 0.0.0.0 --port 8000      (macOS: Core ML)

    POST /scans         multipart: photo (file); plot_id, flight_start, flight_end, scan_id (optional fields)
                        → 202 {"scan_id", "status": "queued"}
    GET  /scans/{id}    → {"scan_id", "status": queued | running | done | failed, "result" when done}
    GET  /health

The Core ML model loads once at startup; scans run one at a time on a single worker thread. Environment:
KROPS_DATA_DIR (uploads, results, weather cache; default integration/data), KROPS_PLOTS (a JSON list of
{plot_id, lat, lon}; default weather/plots.json), KROPS_SETTINGS (a JSON of Settings overrides),
KROPS_NO_WEATHERKIT=1 (NASA POWER only). WeatherKit keys come from weather/.env, as for the weather job.
"""
import json
import logging
import os
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from . import __version__
from .config import REPO, Settings
from .pipeline import run_scan

log = logging.getLogger("krops_vari.api")
SCAN_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
PHOTO_TYPES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def _utc(text, name):
    try:
        moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        raise HTTPException(422, f"{name} must be an ISO date-time in UTC, e.g. 2026-10-26T02:40:00Z")
    if moment.utcoffset() is None:
        raise HTTPException(422, f"{name} needs a time zone (Z for UTC)")
    return text


def create_app(settings=None, data_dir=None, plots_path=None, detector=None, weather=None):
    """The app. Arguments override the environment; tests pass a detector or weather stub."""
    settings = settings or (Settings.from_json(os.environ["KROPS_SETTINGS"]) if os.getenv("KROPS_SETTINGS")
                            else Settings())
    data_dir = Path(data_dir or os.getenv("KROPS_DATA_DIR", REPO / "integration" / "data"))
    plots_path = Path(plots_path or os.getenv("KROPS_PLOTS", REPO / "weather" / "plots.json"))
    scans_dir = data_dir / "scans"
    state = {"detector": detector, "weather": weather}
    jobs, jobs_lock = {}, threading.Lock()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="scan")

    @asynccontextmanager
    async def lifespan(_app):
        if state["detector"] is None:
            from .detection import Detector

            state["detector"] = Detector(settings.model_path)
            log.info("Core ML model loaded")
        yield
        executor.shutdown(wait=True)

    app = FastAPI(title="Krops VARI", version=__version__, lifespan=lifespan)

    def weather_client():
        if state["weather"] is None:
            from .weather import WeatherClient

            state["weather"] = WeatherClient(data_dir / "weather_cache",
                                             use_weatherkit=os.getenv("KROPS_NO_WEATHERKIT") != "1")
        return state["weather"]

    def set_status(scan_id, **fields):
        with jobs_lock:
            jobs[scan_id] = jobs.get(scan_id, {}) | fields

    def work(scan_id, photo_path, plot, flight):
        set_status(scan_id, status="running")
        try:
            result = run_scan(photo_path, settings, detector=state["detector"], scan_id=scan_id, plot=plot,
                              flight=flight, weather=weather_client() if plot else None)
            (scans_dir / scan_id / "result.json").write_text(json.dumps(result, ensure_ascii=False))
            set_status(scan_id, status="done")
        except Exception as error:  # noqa: BLE001 – the client gets the type; the log gets the trace
            log.exception("scan %s failed", scan_id)
            set_status(scan_id, status="failed", error=type(error).__name__)

    @app.get("/health")
    def health():
        return {"status": "ok", "version": __version__, "model_loaded": state["detector"] is not None}

    @app.post("/scans", status_code=202)
    async def create_scan(photo: UploadFile = File(...), plot_id: str | None = Form(None),
                          flight_start: str | None = Form(None), flight_end: str | None = Form(None),
                          scan_id: str | None = Form(None)):
        scan_id = scan_id or uuid.uuid4().hex
        if not SCAN_ID.match(scan_id):
            raise HTTPException(422, "scan_id: 1–64 letters, digits, - or _")
        suffix = Path(photo.filename or "").suffix.lower()
        if suffix not in PHOTO_TYPES:
            raise HTTPException(415, f"photo must be one of {', '.join(sorted(PHOTO_TYPES))}")
        if bool(flight_start) != bool(flight_end):
            raise HTTPException(422, "give both flight_start and flight_end, or neither")
        flight = ({"flight_start": _utc(flight_start, "flight_start"), "flight_end": _utc(flight_end, "flight_end")}
                  if flight_start else None)
        plot = None
        if plot_id:
            plots = json.loads(plots_path.read_text()) if plots_path.is_file() else []
            plot = next((p for p in plots if p["plot_id"] == plot_id), None)
            if plot is None:
                raise HTTPException(404, f"unknown plot_id {plot_id}")
        elif flight:
            raise HTTPException(422, "flight times need a plot_id (the weather is looked up for the plot)")
        folder = scans_dir / scan_id
        with jobs_lock:
            if scan_id in jobs or folder.exists():
                raise HTTPException(409, f"scan {scan_id} already exists")
            jobs[scan_id] = {"status": "queued"}
        folder.mkdir(parents=True)
        photo_path = folder / f"photo{suffix}"
        photo_path.write_bytes(await photo.read())
        executor.submit(work, scan_id, photo_path, plot, flight)
        return {"scan_id": scan_id, "status": "queued"}

    @app.get("/scans/{scan_id}")
    def get_scan(scan_id: str):
        if not SCAN_ID.match(scan_id):
            raise HTTPException(422, "invalid scan_id")
        with jobs_lock:
            job = dict(jobs.get(scan_id, {}))
        result_path = scans_dir / scan_id / "result.json"
        if result_path.is_file():  # also after a restart
            return {"scan_id": scan_id, "status": "done", "result": json.loads(result_path.read_text())}
        if not job:
            raise HTTPException(404, f"unknown scan {scan_id}")
        return JSONResponse({"scan_id": scan_id, **job})

    return app


app = create_app()  # the model loads when the server starts, not on import
