"""The HTTP API: upload dji_10.jpg, poll until done, and get the notebook's zones back; plus input validation.
Needs macOS (Core ML) and fastapi."""
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

from helpers import assert_close
from krops_vari.config import REPO, Settings

FIXTURE = Path(__file__).parent / "fixtures" / "dji_10_notebook.json"
PHOTO = REPO / "segmentation" / "final_flow_ut1" / "dji_10.jpg"


@unittest.skipUnless(sys.platform == "darwin", "Core ML runs on macOS only")
class Api(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            raise unittest.SkipTest("fastapi is not installed (pip install -e 'integration[api]')")
        from krops_vari.api import create_app

        cls.tmp = tempfile.TemporaryDirectory()
        plots = Path(cls.tmp.name) / "plots.json"
        plots.write_text(json.dumps([{"plot_id": "known", "lat": 0.0, "lon": 0.0}]))
        cls.client_cm = TestClient(create_app(Settings(), data_dir=cls.tmp.name, plots_path=plots))
        cls.client = cls.client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_cm.__exit__(None, None, None)
        cls.tmp.cleanup()

    def post(self, data=None, name="dji_10.jpg"):
        with open(PHOTO, "rb") as f:
            return self.client.post("/scans", files={"photo": (name, f, "image/jpeg")}, data=data or {})

    def test_health(self):
        self.assertEqual(self.client.get("/health").json()["model_loaded"], True)

    def test_upload_then_result_equals_the_notebook(self):
        reply = self.post({"scan_id": "golden-1"})
        self.assertEqual(reply.status_code, 202, reply.text)
        for _ in range(120):
            body = self.client.get("/scans/golden-1").json()
            if body["status"] in ("done", "failed"):
                break
            time.sleep(0.5)
        self.assertEqual(body["status"], "done", body)
        assert_close(self, body["result"]["zones"], json.loads(FIXTURE.read_text())["zones"], "zones")
        self.assertEqual(self.post({"scan_id": "golden-1"}).status_code, 409)

    def test_validation(self):
        self.assertEqual(self.post({"flight_start": "2026-10-26T03:00:00Z"}).status_code, 422)  # no end
        self.assertEqual(self.post({"scan_id": "../escape"}).status_code, 422)
        self.assertEqual(self.post({"plot_id": "nope"}).status_code, 404)
        self.assertEqual(self.post(name="notes.txt").status_code, 415)
        self.assertEqual(self.post({"plot_id": "known", "flight_start": "2026-10-26T03:00:00",
                                    "flight_end": "2026-10-26T03:20:00"}).status_code, 422)  # no time zone
        self.assertEqual(self.client.get("/scans/never-made").status_code, 404)


if __name__ == "__main__":
    unittest.main()
