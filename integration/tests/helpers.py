"""Shared test helpers: nested comparison with a float tolerance."""
import math


def assert_close(test, got, want, path="", rel=1e-9):
    """Nested dicts/lists equal, floats to `rel`; the failure names the path to the first difference."""
    if isinstance(want, dict):
        test.assertIsInstance(got, dict, path)
        test.assertEqual(sorted(got), sorted(want), f"keys at {path or 'root'}")
        for key in want:
            assert_close(test, got[key], want[key], f"{path}.{key}", rel)
    elif isinstance(want, list):
        test.assertIsInstance(got, list, path)
        test.assertEqual(len(got), len(want), f"length at {path}")
        for i, (g, w) in enumerate(zip(got, want)):
            assert_close(test, g, w, f"{path}[{i}]", rel)
    elif isinstance(want, float) and not isinstance(got, bool) and isinstance(got, (int, float)):
        test.assertTrue(math.isclose(got, want, rel_tol=rel, abs_tol=1e-12), f"{path}: {got} != {want}")
    else:
        test.assertEqual(got, want, path)
