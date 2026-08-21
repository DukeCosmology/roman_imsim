"""Tests for dispersion-aware stamp trail sizing and placement."""

import math
import os
import unittest

import numpy as np

from roman_imsim.dispersion_trail import (
    DispersionTrail,
    compute_dispersion_trail,
    even_ceil,
    get_dispersive_op_config,
    load_optical_disperser,
    sample_wavelengths,
)
from roman_imsim.photonOps import GrismV, SlitlessSpec, WFSSSDisperser
from roman_imsim.stamp import Roman_stamp


class TestDispersionTrailHelpers(unittest.TestCase):
    def test_even_ceil(self):
        self.assertEqual(even_ceil(0), 2)
        self.assertEqual(even_ceil(3), 4)
        self.assertEqual(even_ceil(4.1), 6)
        self.assertEqual(even_ceil(10), 10)

    def test_get_dispersive_op_config(self):
        cfg = {
            "photon_ops": [
                {"type": "ChargeDiff"},
                {"type": "SlitlessSpec"},
            ]
        }
        op = get_dispersive_op_config(cfg)
        self.assertEqual(op["type"], "SlitlessSpec")
        self.assertIsNone(get_dispersive_op_config({"photon_ops": [{"type": "ChargeDiff"}]}))
        self.assertIsNone(get_dispersive_op_config({}))

    def test_slitless_trail_contains_band_edges(self):
        disperser = SlitlessSpec()
        x0, y0 = 2000.0, 2000.0
        sca = 16
        lam = sample_wavelengths(disperser.wl_min, disperser.wl_max, n=21)
        pad = 32.0
        trail = compute_dispersion_trail(
            disperser.disperse_sca, x0, y0, lam, sca=sca, pad=pad
        )

        # Absolute SCA bounds that the stamp would cover when centered on the trail.
        half_x = trail.xsize / 2.0
        half_y = trail.ysize / 2.0
        stamp_xmin = trail.center_x - half_x + 0.5
        stamp_xmax = trail.center_x + half_x - 0.5
        stamp_ymin = trail.center_y - half_y + 0.5
        stamp_ymax = trail.center_y + half_y - 0.5

        self.assertLessEqual(stamp_xmin, trail.xmin)
        self.assertGreaterEqual(stamp_xmax, trail.xmax)
        self.assertLessEqual(stamp_ymin, trail.ymin)
        self.assertGreaterEqual(stamp_ymax, trail.ymax)
        # Prism toy model is mostly a y shift.
        self.assertGreater(trail.ysize, trail.xsize)
        self.assertEqual(trail.xsize % 2, 0)
        self.assertEqual(trail.ysize % 2, 0)

    def test_optical_disperser_trail_contains_band_edges(self):
        root = os.path.dirname(os.path.dirname(__file__))
        config = os.path.join(root, "optical_models/Roman_prism_OpticalModel_v0.8.yaml")
        optical = load_optical_disperser(config)
        x0, y0 = 2044.5, 2044.5
        sca = 16
        lam = sample_wavelengths(optical.wl_min, optical.wl_max, n=21)
        pad = 40.0
        trail = compute_dispersion_trail(
            optical.disperse_sca, x0, y0, lam, sca=sca, order="1", pad=pad
        )

        half_x = trail.xsize / 2.0
        half_y = trail.ysize / 2.0
        self.assertLessEqual(trail.center_x - half_x + 0.5, np.min(trail.x_disp))
        self.assertGreaterEqual(trail.center_x + half_x - 0.5, np.max(trail.x_disp))
        self.assertLessEqual(trail.center_y - half_y + 0.5, np.min(trail.y_disp))
        self.assertGreaterEqual(trail.center_y + half_y - 0.5, np.max(trail.y_disp))

    def test_wfss_disperser_uses_explicit_sca(self):
        root = os.path.dirname(os.path.dirname(__file__))
        config = os.path.join(root, "optical_models/Roman_prism_OpticalModel_v0.8.yaml")
        op = WFSSSDisperser(config=config, sca=10, order="1")
        self.assertEqual(op.sca, 10)
        xd, yd = op.disperse_sca(
            [2044.5], [2044.5], [op.wl_reference], sca=10, order="1", pairwise=True
        )
        self.assertEqual(np.shape(xd), (1,))
        self.assertEqual(np.shape(yd), (1,))

    def test_grismv_requires_sca(self):
        with self.assertRaises(ValueError):
            GrismV()


class TestRomanStampLocateTrail(unittest.TestCase):
    def test_locate_stamp_centers_on_trail(self):
        import galsim

        builder = Roman_stamp()
        # Fake a trail displaced from the object.
        trail = DispersionTrail(
            xmin=2100.0,
            xmax=2300.0,
            ymin=1800.0,
            ymax=2200.0,
            center_x=2200.0,
            center_y=2000.0,
            xsize=240,
            ysize=440,
            wavelengths=np.array([1.0, 1.5]),
            x_disp=np.array([2100.0, 2300.0]),
            y_disp=np.array([1800.0, 2200.0]),
        )
        builder._dispersion_trail = trail

        image_pos = galsim.PositionD(2000.3, 1999.7)
        base = {
            "wcs": galsim.PixelScale(0.11),
            "image_center": galsim.PositionD(2044.0, 2044.0),
        }
        config = {}
        logger = galsim.config.LoggerWrapper(None)
        builder.locateStamp(
            config, base, trail.xsize, trail.ysize, image_pos, None, logger
        )

        self.assertIsNotNone(base["stamp_center"])
        # Stamp center should be near the trail midpoint (even-size +0.5 convention).
        self.assertAlmostEqual(base["stamp_center"].x, 2200, delta=1)
        self.assertAlmostEqual(base["stamp_center"].y, 2000, delta=1)

        # Object draw location = stamp_center + stamp_offset ≈ nominal undispersed.
        nominal_x = image_pos.x + 0.5  # even xsize
        nominal_y = image_pos.y + 0.5
        drawn_x = base["stamp_center"].x + base["stamp_offset"].x
        drawn_y = base["stamp_center"].y + base["stamp_offset"].y
        self.assertAlmostEqual(drawn_x, nominal_x, places=6)
        self.assertAlmostEqual(drawn_y, nominal_y, places=6)

        # Dispersed endpoints remain inside the stamp bounds that makeStamp would build.
        bounds = galsim.BoundsI(1, trail.xsize, 1, trail.ysize)
        bounds = bounds.shift(base["stamp_center"] - bounds.center)
        for x, y in zip(trail.x_disp, trail.y_disp):
            self.assertTrue(bounds.xmin - 0.5 <= x <= bounds.xmax + 0.5)
            self.assertTrue(bounds.ymin - 0.5 <= y <= bounds.ymax + 0.5)


if __name__ == "__main__":
    unittest.main()
