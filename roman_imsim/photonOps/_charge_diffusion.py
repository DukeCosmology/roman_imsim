import numpy as np
from galsim import GaussianDeviate, PhotonOp, UniformDeviate
from galsim.config import GetAllParams, GetRNG, PhotonOpBuilder, RegisterPhotonOpType, get_cls_params

__all__ = ["ChargeDiff"]

_w1 = 0.17519
_w2 = 0.53146
_w3 = 0.29335
_s = 0.3279
_s1 = 0.4522 * _s
_s2 = 0.8050 * _s
_s3 = 1.4329 * _s


class ChargeDiff(PhotonOp):
    """A photon operator that applies the effect of charge diffusion via a probablistic model limit."""

    def __init__(self, rng=None, **kwargs):

        self.rng = rng

    def applyTo(self, photon_array, local_wcs=None, rng=None):
        """Apply the charge diffusion effect to the photons

        Parameters:
            photon_array:   A `PhotonArray` to apply the operator to.
            local_wcs:      A `LocalWCS` instance defining the local WCS for the current photon
                            bundle in case the operator needs this information.  [default: None]
            rng:            A random number generator to use if needed. [default: None]
        """

        self.ud = UniformDeviate(rng)
        self.gd1 = GaussianDeviate(rng, sigma=_s1)
        self.gd2 = GaussianDeviate(rng, sigma=_s2)
        self.gd3 = GaussianDeviate(rng, sigma=_s3)

        # Choose which weighted Gausian to use in sech model approximation
        u = np.empty(len(photon_array.x))
        self.ud.generate(u)

        # Selects appropriate fraction of photons corresponding to the first gaussian in the sech model
        mask = u < _w1
        dx = np.empty(np.sum(mask))
        dy = np.empty(np.sum(mask))
        # Generate and apply the 2D gaussian shifts corresponding to the first gaussian
        self.gd1.generate(dx)
        self.gd1.generate(dy)
        photon_array.x[mask] += dx
        photon_array.y[mask] += dy

        # Selects appropriate fraction of photons corresponding to the second gaussian in the sech model
        mask = (u >= _w1) & (u <= (1.0 - _w3))
        dx = np.empty(np.sum(mask))
        dy = np.empty(np.sum(mask))
        # Generate and apply the 2D gaussian shifts corresponding to the second gaussian
        self.gd2.generate(dx)
        self.gd2.generate(dy)
        photon_array.x[mask] += dx
        photon_array.y[mask] += dy

        # Selects appropriate fraction of photons corresponding to the third gaussian in the sech model
        mask = u > (1.0 - _w3)
        dx = np.empty(np.sum(mask))
        dy = np.empty(np.sum(mask))
        # Generate and apply the 2D gaussian shifts corresponding to the second gaussian
        self.gd3.generate(dx)
        self.gd3.generate(dy)
        photon_array.x[mask] += dx
        photon_array.y[mask] += dy


class ChargeDiffBuilder(PhotonOpBuilder):
    """Build ChargeDiff photonOp"""

    def buildPhotonOp(self, config, base, logger):
        req, opt, single, takes_rng = get_cls_params(ChargeDiff)
        kwargs, safe = GetAllParams(config, base, req, opt, single)
        rng = GetRNG(config, base, logger, "Roman_stamp")
        kwargs["rng"] = rng
        return ChargeDiff(**kwargs)


RegisterPhotonOpType("ChargeDiff", ChargeDiffBuilder())
