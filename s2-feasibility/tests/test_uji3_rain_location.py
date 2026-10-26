"""Uji 3 takes each plot's rain from its own centroid, never from the .env point: s2_rain.rain_by_plot.

    cd s2-feasibility && python -m unittest discover -s tests -v

The fetchers are fakes that record where they were asked and answer with rain that depends on the place, so a plot
fed another place's rain would show the wrong P30. No network.
"""
import sys
import unittest
from datetime import date, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
try:
    from shapely.geometry import box

    import s2_rain
    from s2_rain import job
    import l2_scan_rain as l2
except ImportError:  # shapely, or requests for the weather job
    s2_rain = None

needs_libs = unittest.skipUnless(s2_rain, "shapely or requests not installed (conda env create -f environment.yml)")

ENV_POINT = (0.92, 104.45)  # stands in for the .env point: no plot below is in its cell
END = date(2024, 3, 31)
SCAN = date(2024, 3, 15)


def plot(plot_id, lat, lon, size=0.002):
    return {"plot_id": plot_id, "geometry": box(lon, lat, lon + size, lat + size)}


class FakeWeather:
    """Daily rain by latitude: 1 mm/day south of the equator (dry), 5 mm/day north (wet). Records every call."""

    def __init__(self, wk=True):
        self.nasa_calls, self.wk_calls, self.wk = [], [], wk

    @staticmethod
    def _rain(lat, start, end):
        mm = 1.0 if lat < 0 else 5.0
        return {start + timedelta(days=i): mm for i in range((end - start).days + 1)}

    def nasa(self, lat, lon, start, end):
        self.nasa_calls.append((lat, lon))
        return self._rain(lat, start, end)

    def weatherkit(self, lat, lon, start, end):
        self.wk_calls.append((lat, lon))
        return self._rain(lat, start, end) if self.wk else None


@needs_libs
class TwoPlotsFarApart(unittest.TestCase):
    """One plot in Riau, one in West Java: about 900 km apart, different cells, different rain."""

    def setUp(self):
        self.plots = [plot("riau", 0.50, 101.40), plot("jabar", -6.90, 107.60)]
        self.fake = FakeWeather()
        self.sources, self.counts = s2_rain.rain_by_plot(self.plots, END, self.fake.nasa, self.fake.weatherkit)

    def expected(self, p):
        lat, lon = s2_rain.centroid(p)
        return job.grid_cell(lat, lon), job.weatherkit_area(lat, lon)

    def test_each_plot_fetched_at_its_own_cell_and_area(self):
        cells = {(c["lat"], c["lon"]) for c, _ in map(self.expected, self.plots)}
        areas = {(a["lat"], a["lon"]) for _, a in map(self.expected, self.plots)}
        self.assertEqual(set(self.fake.nasa_calls), cells)
        self.assertEqual(set(self.fake.wk_calls), cells | areas)  # cell centre for the monthly check, area for P30
        self.assertEqual((self.counts["nasa_cells"], self.counts["weatherkit_areas"]), (2, 2))

    def test_the_env_point_is_never_asked(self):
        env_cell = job.grid_cell(*ENV_POINT)
        env_area = job.weatherkit_area(*ENV_POINT)
        asked = set(self.fake.nasa_calls) | set(self.fake.wk_calls)
        self.assertNotIn((env_cell["lat"], env_cell["lon"]), asked)
        self.assertNotIn((env_area["lat"], env_area["lon"]), asked)

    def test_p30_and_water_stress_follow_the_plot(self):
        riau = l2.p30_at(self.sources["riau"], SCAN)
        jabar = l2.p30_at(self.sources["jabar"], SCAN)
        self.assertEqual((riau["p30_mm"], riau["p30_source"], riau["water_stress"]), (150.0, "weatherkit-history", False))
        self.assertEqual((jabar["p30_mm"], jabar["p30_source"], jabar["water_stress"]), (30.0, "weatherkit-history", True))

    def test_nasa_fallback_is_the_plots_own_cell(self):
        fake = FakeWeather(wk=False)
        sources, counts = s2_rain.rain_by_plot(self.plots, END, fake.nasa, fake.weatherkit)
        self.assertFalse(counts["weatherkit"])
        self.assertEqual(l2.p30_at(sources["riau"], SCAN)["p30_mm"], 150.0)
        self.assertEqual(l2.p30_at(sources["jabar"], SCAN)["p30_mm"], 30.0)
        self.assertEqual(l2.p30_at(sources["jabar"], SCAN)["p30_source"], "nasa-power")


@needs_libs
class NearbyPlotsShare(unittest.TestCase):
    def test_one_fetch_per_cell_and_area(self):
        plots = [plot("a", 0.51, 101.41), plot("b", 0.52, 101.42)]  # about 1.5 km apart, same cell and area
        fake = FakeWeather()
        _, counts = s2_rain.rain_by_plot(plots, END, fake.nasa, fake.weatherkit)
        self.assertEqual((len(fake.nasa_calls), counts["nasa_cells"], counts["weatherkit_areas"]), (1, 1, 1))


@needs_libs
class Privacy(unittest.TestCase):
    def test_failed_request_hides_the_url(self):
        def boom():
            raise RuntimeError("GET https://power.larc.nasa.gov/...?latitude=0.5&longitude=101.4 failed")

        with self.assertRaises(s2_rain.RainFetchFailed) as ctx:
            s2_rain._quiet(boom)
        self.assertEqual(str(ctx.exception), "RuntimeError")

    def test_job_logger_silenced_while_fetching(self):
        import logging

        logger = logging.getLogger("nasa_power_job")
        seen = s2_rain._quiet(lambda: logger.disabled)
        self.assertTrue(seen)
        self.assertFalse(logger.disabled)


if __name__ == "__main__":
    unittest.main()
