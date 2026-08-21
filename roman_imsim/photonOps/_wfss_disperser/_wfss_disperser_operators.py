"""WFSS disperser PhotonOp backed by GDPS or in-tree SNPIT optical models."""

from galsim import GalSimError, PhotonOp
from galsim.config import (
    GetAllParams,
    ParseValue,
    PhotonOpBuilder,
    RegisterPhotonOpType,
    get_cls_params,
)

from roman_imsim.dispersion_trail import load_optical_disperser, resolve_optical_model_path

__all__ = ["WFSSSDisperser"]


class WFSSSDisperser(PhotonOp):
    """A photon operator that applies Roman prism/grism dispersion.

    Uses ``roman_gdps_optical_model.RomanOpticalModel`` when available and the
    config uses the nested GDPS layout; otherwise uses the in-tree
    ``SNPITDisperser``. Both share the same ``disperse_sca`` interface used for
    stamp trail sizing.

    Parameters:
        config:    Path to the YAML optical model config file.
        order:     Spectral order key (e.g. ``'1'``).
        sca:       SCA number.
    """

    _req_params = {}
    _opt_params = {
        "config": str,
        "order": str,
        "sca": int,
    }
    _single_params = []
    _takes_rng = False

    def __init__(self, config=None, order=None, sca=None):
        self.config = resolve_optical_model_path(
            config or "optical_models/Roman_prism_OpticalModel_v0.8.yaml"
        )
        self.order = order or "1"
        if sca is None:
            raise ValueError("WFSSSDisperser requires an explicit sca (use image.SCA)")
        self.sca = int(sca)

        self._optical = load_optical_disperser(self.config)
        # Keep attribute name used by older tests / callers.
        self.snpit_disperser = getattr(self._optical, "_disperser", None)
        self.wl_min = self._optical.wl_min
        self.wl_max = self._optical.wl_max
        self.wl_reference = self._optical.wl_reference
        self.backend = self._optical.backend

    def disperse_sca(self, x0, y0, lam, sca=None, order=None, pairwise=False):
        """Disperse undispersed SCA coordinates; used by PhotonOp and stamp sizing."""
        return self._optical.disperse_sca(
            x0,
            y0,
            lam,
            sca if sca is not None else self.sca,
            order=order if order is not None else self.order,
            pairwise=pairwise,
        )

    def applyTo(self, photon_array, local_wcs=None, rng=None):
        """Apply the dispersion to the photons.

        Parameters:
            photon_array:   A `PhotonArray` to apply the operator to.
            local_wcs:      A `LocalWCS` instance defining the local WCS for the current photon
                            bundle in case the operator needs this information.  [default: None]
            rng:            A random number generator is not used.
        """
        if not photon_array.hasAllocatedWavelengths():
            raise GalSimError("WFSSSDisperser requires that wavelengths be set")

        # wavelength is in nm. Optical model expects microns.
        w = photon_array.wavelength / 1000.0
        x_pix, y_pix = self.disperse_sca(
            photon_array.x,
            photon_array.y,
            w,
            self.sca,
            self.order,
            pairwise=True,
        )
        photon_array.x = x_pix
        photon_array.y = y_pix

    def __repr__(self):
        return (
            f"roman_imsim.WFSSSDisperser(config={self.config!r}, "
            f"order={self.order!r}, sca={self.sca})"
        )


class WFSSSDisperserBuilder(PhotonOpBuilder):
    """Build a WFSSSDisperser, defaulting SCA from ``image.SCA``."""

    def buildPhotonOp(self, config, base, logger):
        from galsim import GalSimConfigError

        req, opt, single, takes_rng = get_cls_params(WFSSSDisperser)
        kwargs, safe = GetAllParams(config, base, req, opt, single)
        if "sca" not in kwargs:
            if "SCA" not in base.get("image", {}):
                raise GalSimConfigError(
                    "WFSSSDisperser requires stamp.photon_ops.sca or image.SCA"
                )
            kwargs["sca"] = ParseValue(base["image"], "SCA", base, int)[0]
        return WFSSSDisperser(**kwargs)


RegisterPhotonOpType("WFSSSDisperser", WFSSSDisperserBuilder())
