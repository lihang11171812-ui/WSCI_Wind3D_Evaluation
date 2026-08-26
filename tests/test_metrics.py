from pathlib import Path
import tempfile
import unittest

import numpy as np

from wind_eval.coherence import evaluate_coherence
from wind_eval.wsci import evaluate_wsci


def make_data(path: Path, nt=5):
    z, y, x = 6, 18, 18
    xx, yy = np.meshgrid(np.arange(x), np.arange(y))
    heights = np.arange(1, z + 1) * 20.0
    u = np.empty((nt, z, y, x)); v = np.empty_like(u); w = np.empty_like(u)
    dem = 2.0 * np.sin(xx / 5.0)
    dHdy, dHdx = np.gradient(dem, 30.0, 30.0)
    for t in range(nt):
        for k, h in enumerate(heights):
            u[t, k] = 3.0 * (h / 100.0) ** 0.2 + 0.02 * xx + 0.01 * t
            v[t, k] = 0.5 + 0.01 * yy
            w[t, k] = u[t, k] * dHdx + v[t, k] * dHdy
    np.savez(path, u=u, v=v, w=w, dem_m=dem,
             z_abs_m=heights, dx_m=30.0, dy_m=30.0)


class MetricTests(unittest.TestCase):
    def wind_file(self, nt=5):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "wind.npz"
        make_data(path, nt=nt)
        return path

    def test_scores_are_bounded(self):
        p = self.wind_file()
        a, b = evaluate_wsci(p), evaluate_coherence(p, dt=10.0)
        self.assertTrue(0 <= a["total_0_1"] <= 1)
        self.assertTrue(0 <= b["coherence_0_1"] <= 1)

    def test_missing_true_is_not_perfect(self):
        result = evaluate_wsci(self.wind_file())
        self.assertIsNone(result["components"]["Is"]["P_var"])
        self.assertNotIn("P_var", result["components"]["Is"]["effective_weights"])

    def test_single_frame_excludes_time(self):
        result = evaluate_coherence(self.wind_file(nt=1))
        self.assertIsNone(result["components"]["time"]["score"])
        self.assertNotIn("time", result["effective_weights"])


if __name__ == "__main__":
    unittest.main()
