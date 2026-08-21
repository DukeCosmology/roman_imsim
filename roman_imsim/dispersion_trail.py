"""Helpers to size and center stamps on dispersed spectral trails.

The imaging stamp path centers on the undispersed WCS position. For prism/grism
PhotonOps, photons are relocated onto a spectral trail after shooting, so the
stamp must cover that trail. These helpers evaluate the dispersion model at a
wavelength grid and return trail extents / center for ``Roman_stamp``.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import yaml

__all__ = [
    "DispersionTrail",
    "DISPERSIVE_PHOTON_OP_TYPES",
    "compute_dispersion_trail",
    "even_ceil",
    "get_dispersive_op_config",
    "load_optical_disperser",
    "resolve_optical_model_path",
    "sample_wavelengths",
]


DISPERSIVE_PHOTON_OP_TYPES = frozenset(
    {"SlitlessSpec", "WFSSSDisperser", "GrismV", "GrismNV"}
)


@dataclass(frozen=True)
class DispersionTrail:
    """Bounding box of a dispersed spectral trail on the SCA."""

    xmin: float
    xmax: float
    ymin: float
    ymax: float
    center_x: float
    center_y: float
    xsize: int
    ysize: int
    wavelengths: np.ndarray
    x_disp: np.ndarray
    y_disp: np.ndarray


def even_ceil(n: float) -> int:
    """Return the smallest even integer >= max(ceil(n), 2)."""
    n = int(math.ceil(n))
    if n < 2:
        n = 2
    if n % 2:
        n += 1
    return n


def resolve_optical_model_path(config_file: Optional[str]) -> str:
    """Resolve an optical-model YAML path relative to the roman_imsim repo root."""
    if config_file is None:
        config_file = "optical_models/Roman_prism_OpticalModel_v0.8.yaml"
    if os.path.isabs(config_file) and os.path.isfile(config_file):
        return config_file
    # roman_imsim/roman_imsim/dispersion_trail.py -> repo root
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidate = os.path.join(repo_root, config_file)
    if os.path.isfile(candidate):
        return candidate
    if os.path.isfile(config_file):
        return os.path.abspath(config_file)
    raise FileNotFoundError(f"Optical model config not found: {config_file}")


class _SNPITOpticalDisperser:
    """Thin adapter around SNPITDisperser with wavelength metadata."""

    def __init__(self, config_file: str):
        from roman_imsim.photonOps._wfss_disperser.snpitdispenser import (
            SNPITDisperser,
        )

        self.config_file = config_file
        self._disperser = SNPITDisperser(config_file)
        self.wl_min = float(self._disperser.optical["wl_min"])
        self.wl_max = float(self._disperser.optical["wl_max"])
        self.wl_reference = float(self._disperser.optical["wl_reference"])
        self.backend = "snpit"

    def disperse_sca(self, x0, y0, lam, sca, order="1", pairwise=False):
        return self._disperser.disperse(
            x0, y0, lam, sca, order=order, pairwise=pairwise
        )


class _GDPSOpticalDisperser:
    """Adapter around roman_gdps_optical_model.RomanOpticalModel."""

    def __init__(self, config_file: str):
        from roman_gdps_optical_model import RomanOpticalModel

        self.config_file = config_file
        self._model = RomanOpticalModel(config_file)
        self.wl_min = float(self._model.wl_min)
        self.wl_max = float(self._model.wl_max)
        self.wl_reference = float(self._model.wl_reference)
        self.backend = "gdps"

    def disperse_sca(self, x0, y0, lam, sca, order="1", pairwise=False):
        return self._model.disperse_sca(
            x0, y0, lam, sca, order=order, pairwise=pairwise
        )


def load_optical_disperser(config_file: Optional[str] = None):
    """Load a GDPS or SNPIT optical disperser for the given YAML config.

    Prefers ``roman_gdps_optical_model.RomanOpticalModel`` when the package is
    importable and the YAML uses the nested ``roman:`` GDPS layout; otherwise
    falls back to the in-tree ``SNPITDisperser``.
    """
    path = resolve_optical_model_path(config_file)
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if isinstance(raw, dict) and "roman" in raw:
        try:
            return _GDPSOpticalDisperser(path)
        except ImportError:
            pass

    return _SNPITOpticalDisperser(path)


def sample_wavelengths(
    wl_min: float,
    wl_max: float,
    n: int = 21,
    bandpass=None,
) -> np.ndarray:
    """Sample wavelengths in microns, optionally clipped to a GalSim bandpass."""
    if bandpass is not None:
        # GalSim Bandpass limits are in nm.
        bmin = float(bandpass.blue_limit) / 1000.0
        bmax = float(bandpass.red_limit) / 1000.0
        wl_min = max(wl_min, bmin)
        wl_max = min(wl_max, bmax)
    if not np.isfinite(wl_min) or not np.isfinite(wl_max) or wl_max <= wl_min:
        raise ValueError(f"Invalid wavelength range: [{wl_min}, {wl_max}]")
    return np.linspace(wl_min, wl_max, int(n))


def compute_dispersion_trail(
    disperse_fn,
    x0: float,
    y0: float,
    wavelengths: Sequence[float],
    sca: int,
    order: str = "1",
    pad: float = 0.0,
) -> DispersionTrail:
    """Evaluate ``disperse_fn`` along ``wavelengths`` and return a padded trail.

    Parameters
    ----------
    disperse_fn :
        Callable ``(x0, y0, lam, sca, order, pairwise=True) -> (xd, yd)``.
    x0, y0 :
        Undispersed SCA pixel coordinates.
    wavelengths :
        Physical wavelengths in microns.
    sca :
        SCA index.
    order :
        Spectral order key.
    pad :
        Extra pixels of morphology/PSF padding on each side.
    """
    lam = np.asarray(wavelengths, dtype=np.float64).ravel()
    if lam.size < 2:
        raise ValueError("Need at least two wavelengths to define a trail")

    x_in = np.full(lam.shape, float(x0), dtype=np.float64)
    y_in = np.full(lam.shape, float(y0), dtype=np.float64)
    xd, yd = disperse_fn(x_in, y_in, lam, int(sca), order, pairwise=True)
    xd = np.asarray(xd, dtype=np.float64).ravel()
    yd = np.asarray(yd, dtype=np.float64).ravel()

    xmin = float(np.min(xd))
    xmax = float(np.max(xd))
    ymin = float(np.min(yd))
    ymax = float(np.max(yd))
    pad = float(pad)
    xsize = even_ceil((xmax - xmin) + 2.0 * pad)
    ysize = even_ceil((ymax - ymin) + 2.0 * pad)

    return DispersionTrail(
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        center_x=0.5 * (xmin + xmax),
        center_y=0.5 * (ymin + ymax),
        xsize=xsize,
        ysize=ysize,
        wavelengths=lam,
        x_disp=xd,
        y_disp=yd,
    )


def get_dispersive_op_config(stamp_config) -> Optional[dict]:
    """Return the first dispersive photon_op dict from stamp config, if any."""
    photon_ops = stamp_config.get("photon_ops")
    if not photon_ops:
        return None
    for op in photon_ops:
        if isinstance(op, dict) and op.get("type") in DISPERSIVE_PHOTON_OP_TYPES:
            return op
    return None


def parse_sca_from_config(op_config: dict, base: dict, logger=None) -> int:
    """Resolve SCA from the photon-op config or ``image.SCA``."""
    import galsim.config

    if op_config is not None and "sca" in op_config:
        return int(galsim.config.ParseValue(op_config, "sca", base, int)[0])
    image = base.get("image", {})
    if "SCA" in image:
        return int(galsim.config.ParseValue(image, "SCA", base, int)[0])
    if logger is not None:
        logger.warning("No SCA found in config; defaulting dispersion SCA to 16")
    return 16
