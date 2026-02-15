from galsim import PhotonOp
from galsim.config import PhotonOpBuilder, RegisterPhotonOpType, get_cls_params, GetAllParams, GetRNG

__all__ = ['SlitlessSpec']

class SlitlessSpec(PhotonOp):
    r"""A photon operator that applies the dispersion effects of the
    Roman Prism.
    
    The photons will need to have wavelengths defined in order to work.
        
    Parameters:
        base_wavelength:    Wavelength (in nm) represented by the fiducial photon positions
    """
    # what parameters are tunable
    # _req_params = {"base_wavelength": float, "barycenter": list}
    # _opt_params = {"resolution": list}

    
    def __init__(self):
        # self.base_wavelength = base_wavelength
        # self.resolution = np.array(resolution)
        pass
    
    def applyTo(self, photon_array, local_wcs=None, rng=None):
        """Apply the slitless-spectroscopy disspersion to the photos
    
        Parameters:
            photon_array:   A `PhotonArray` to apply the operator to.
            local_wcs:      A `LocalWCS` instance defining the local WCS for the current photon
                            bundle in case the operator needs this information.  [default: None]
            rng:            A random number generator is not used.
        """
        #photon array has .x, .y, .wavelength, .coord, .time, ...
        if not photon_array.hasAllocatedWavelengths():
            raise GalSimError("SlitlessSpec requires that wavelengths be set")
        
        # wavelength is in nm. Roman slitless thinks in microns.
        # http://galsim-developers.github.io/GalSim/_build/html/photon_array.html#galsim.PhotonArray
        w = photon_array.wavelength/1000.

        # dx = (-12.973976 + 213.353667*(w - 1.0) + -20.254574*(w - 1.0)**2)/(1.0 + 1.086448*(w - 1.0) + -0.573796*(w - 1.0)**2)
        dy = (-81.993865 + 138.367237*(w - 1.0) + 19.348549*(w - 1.0)**2)/(1.0 + 1.086447*(w - 1.0) + -0.573797*(w - 1.0)**2)
        
        photon_array.y += dy
    
        # might need to change dxdz/dydz for the angle of travel through the detector.
    
    def __repr__(self):
        # s = "galsim.SlitlessSpec(base_wavelength=%r, " % (
        #     self.base_wavelength,
        # )
        # s += ")"
        s = "galsim.SlitlessSpec()"
        return s

class SlitlessSpecBuilder(PhotonOpBuilder):
    """Build a SlitlessSpec
    """
    # This one needs special handling for obj_coord
    def buildPhotonOp(self, config, base, logger):
        req, opt, single, takes_rng = get_cls_params(SlitlessSpec)
        kwargs, safe = GetAllParams(config, base, req, opt, single)
        #if 'sky_pos' in base:
        #    kwargs['obj_coord'] = base['sky_pos']
        return SlitlessSpec(**kwargs)

RegisterPhotonOpType('SlitlessSpec', SlitlessSpecBuilder())