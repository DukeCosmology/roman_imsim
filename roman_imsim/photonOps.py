import numpy as np
from galsim import GaussianDeviate, PhotonOp, UniformDeviate
from galsim.config import (
    GetAllParams,
    GetRNG,
    PhotonOpBuilder,
    RegisterPhotonOpType,
    get_cls_params,
)

_w1 = 0.17519
_w2 = 0.53146
_w3 = 0.29335
_s = 0.3279
_s1 = 0.4522 * _s
_s2 = 0.8050 * _s
_s3 = 1.4329 * _s


class ChargeDiff(PhotonOp):
    """A photon operator that applies the effect of charge diffusion via a
    probabilistic model limit.
    """

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




class BrighterFatter(PhotonOp):
    """
    Gradient-based brighter-fatter PhotonOp.

    The current pre-readout image defines the displacement field.
    Incoming photons are shifted in place.
    """

    def __init__(self, image=None, bfe_strength=1.038e-6, **kwargs):
        self.bfe_strength = float(bfe_strength)
        self.image = None
        self.grad_x = None
        self.grad_y = None

        if image is not None:
            self.set_image(image)

    def set_image(self, image):
        """
        Set the current pre-readout image used to define the BFE field.
        """
        self.image = np.asarray(image, dtype=float)

        # np.gradient returns derivatives in axis order:
        # first output is dI/dy, second output is dI/dx
        self.grad_y, self.grad_x = np.gradient(self.image)

    def applyTo(self, photon_array, local_wcs=None, rng=None):
        """
        Apply brighter-fatter shifts to the incoming photon array in place.
        """
        if self.image is None:
            raise RuntimeError("BrighterFatter requires image to be set via set_image().")

        x = photon_array.x
        y = photon_array.y

        ny, nx = self.image.shape

        ix = np.floor(x).astype(int)
        iy = np.floor(y).astype(int)

        valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)

        x[valid] += -self.bfe_strength * self.grad_x[iy[valid], ix[valid]]
        y[valid] += -self.bfe_strength * self.grad_y[iy[valid], ix[valid]]
        
class BrighterFatterBuilder(PhotonOpBuilder):
    """Build BrighterFatter photonOp"""

    def buildPhotonOp(self, config, base, logger):
        req, opt, single, takes_rng = get_cls_params(BrighterFatter)
        kwargs, safe = GetAllParams(config, base, req, opt, single)
        return BrighterFatter(**kwargs)


RegisterPhotonOpType("BrighterFatter", BrighterFatterBuilder())