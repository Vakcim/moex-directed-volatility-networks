from __future__ import annotations

import numpy as np
import unittest

from v10_common import calibration_scale, qlike


class LossMathTests(unittest.TestCase):
    def test_qlike_is_minimized_at_realized_variance_pointwise(self) -> None:
        y = np.array([0.25, 1.0, 4.0])
        optimum = qlike(y, y)
        low = qlike(y, y / 2)
        high = qlike(y, y * 2)
        self.assertTrue(np.all(optimum < low))
        self.assertTrue(np.all(optimum < high))

    def test_training_calibration_scale_is_positive_and_finite(self) -> None:
        y = np.array([1.0, 2.0, 4.0])
        raw = np.array([0.5, 1.0, 2.0])
        scale = calibration_scale(y, raw)
        self.assertTrue(np.isfinite(scale))
        self.assertGreater(scale, 0)
        self.assertTrue(np.isclose(scale, 2.0))
