import numpy as np

from galsim import GalSimError, PhotonOp
from galsim.config import (
    GetAllParams,
    PhotonOpBuilder,
    RegisterPhotonOpType,
    get_cls_params,
)

__all__ = ["SlitlessSpec"]


def _slitless_dy_um(w_um):
    """Toy prism y-dispersion (pixels) for wavelength in microns."""
    w = np.asarray(w_um, dtype=np.float64)
    return (-81.993865 + 138.367237 * (w - 1.0) + 19.348549 * (w - 1.0) ** 2) / (
        1.0 + 1.086447 * (w - 1.0) + -0.573797 * (w - 1.0) ** 2
    )


class SlitlessSpec(PhotonOp):
    r"""A photon operator that applies the dispersion effects of the
    Roman Prism (toy relative-shift model).

    The photons will need to have wavelengths defined in order to work.
    """

    # Prism-like wavelength coverage used for stamp trail sizing (µm).
    wl_min = 0.75
    wl_max = 1.85
    wl_reference = 1.0

    _req_params = {}
    _opt_params = {}
    _single_params = []
    _takes_rng = False

    def __init__(self):
        pass

    def disperse_sca(self, x0, y0, lam, sca=None, order="1", pairwise=False):
        """Return dispersed SCA coords for trail sizing / consistency checks.

        This toy model only shifts in y: ``y' = y + dy(λ)``.
        """
        x0 = np.atleast_1d(np.asarray(x0, dtype=np.float64))
        y0 = np.atleast_1d(np.asarray(y0, dtype=np.float64))
        lam = np.atleast_1d(np.asarray(lam, dtype=np.float64))
        dy = _slitless_dy_um(lam)

        if pairwise:
            if not (x0.shape == y0.shape == lam.shape):
                raise ValueError("pairwise disperse_sca requires matching shapes")
            return x0.copy(), y0 + dy

        xd = x0[:, np.newaxis] + np.zeros((1, lam.size), dtype=np.float64)
        yd = y0[:, np.newaxis] + dy[np.newaxis, :]
        return xd, yd

    def applyTo(self, photon_array, local_wcs=None, rng=None):
        """Apply the slitless-spectroscopy dispersion to the photons.

        Parameters:
            photon_array:   A `PhotonArray` to apply the operator to.
            local_wcs:      A `LocalWCS` instance defining the local WCS for the current photon
                            bundle in case the operator needs this information.  [default: None]
            rng:            A random number generator is not used.
        """
        if not photon_array.hasAllocatedWavelengths():
            raise GalSimError("SlitlessSpec requires that wavelengths be set")

        # wavelength is in nm. Roman slitless thinks in microns.
        w = photon_array.wavelength / 1000.0
        photon_array.y += _slitless_dy_um(w)

    def __repr__(self):
        return "galsim.SlitlessSpec()"


class SlitlessSpecBuilder(PhotonOpBuilder):
    """Build a SlitlessSpec"""

    def buildPhotonOp(self, config, base, logger):
        req, opt, single, takes_rng = get_cls_params(SlitlessSpec)
        kwargs, safe = GetAllParams(config, base, req, opt, single)
        return SlitlessSpec(**kwargs)


RegisterPhotonOpType("SlitlessSpec", SlitlessSpecBuilder())
