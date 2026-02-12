import numpy as np
import yaml

from galsim import PhotonOp
from galsim.config import PhotonOpBuilder, RegisterPhotonOpType, get_cls_params, GetAllParams, GetRNG
from .snpitdispenser import SNPITDisperser

__all__ = ['WFSSSDisperser']


class WFSSSDisperser(PhotonOp):
    """A photon operator that applies the dispersion effects of the
    Roman Grism (vector version)
    
    The photons will need to have wavelengths defined in order to work.
        
    Parameters:
        config_file:    Path to the YAML config file with optical model
        order:          Grism order (e.g., '1')
        sca:            SCA number
    """
    # what parameters are tunable
    _req_params = {}
    _opt_params = {}
    _single_params = []
    _takes_rng = False
    
    def __init__(self, config=None, order=None, sca=None):
        if config is None:
            self.config = "optical_models/Roman_prism_OpticalModel_v0.8.yaml"
        else:
            self.config = config
        self.order = '1'
        self.sca = sca or 16  # TODO: This is no good.
        # # self.base_wavelength = base_wavelength
        # # self.resolution = np.array(resolution)
        # with open(self.config) as f:
        #     data = yaml.safe_load(f)
        # for order_key, order_data in data["optical_model"]["orders"].items():
        #     if order_key == self.order:
        #         order_dat = order_data
        #         break

        # self.xmap_coeff = np.array(order_dat['xmap_ij_coeff'])
        # self.ymap_coeff = np.array(order_dat['ymap_ij_coeff'])
        # self.crv_coeff = np.array(order_dat['crv_ijk_coeff'])
        # self.ids_coeff = np.array(order_dat['ids_ijk_coeff'])
        # self.wl_min = 0.9 
        # self.wl_max = 2.0
        # self.wl_ref = 1.55
        # # initialize RomanDetectorCoordinates
        # # need these for some of the functions called
        # self.detector_coords = RomanDetectorCoordinates(
        #     naxis1=data['detector_model'].get('naxis1', 4096),  # Adjust these defaults as needed
        #     naxis2=data['detector_model'].get('naxis2', 4096),
        #     crpix1=data['detector_model'].get('crpix1', 2048),
        #     crpix2=data['detector_model'].get('crpix2', 2048),
        #     pos_angle_detector=data['detector_model'].get('pos_angle_detector', 0.0),
        #     pixel_scale=data['detector_model'].get('pixel_scale', 0.11),
        #     plate_scale=data['detector_model'].get('plate_scale', 100.0),
        #     xy_centers=data['detector_model'].get('xy_centers', {})  # Dictionary of SCA centers
        # )
        self.snpit_disperser = SNPITDisperser(self.config)

    def _disperse(self, x0, y0, lam, sca, order='1'):
        # get the coords in degrees to match the matrices
        xfpa_deg, yfpa_deg = self.detector_coords.convert_sca_to_fpa(x0, y0, sca = self.sca)

        # we start by calculating the intial offset
        xfpa_deg = np.atleast_1d(xfpa_deg)
        yfpa_deg = np.atleast_1d(yfpa_deg)
        # Create power matrices
        x_powers = xfpa_deg[:, np.newaxis] ** np.arange(6)  
        y_powers = yfpa_deg[:, np.newaxis] ** np.arange(6)  
        # Compute the offset in mm for x
        xref_mm = np.diagonal(x_powers @ self.xmap_coeff @ y_powers.T)
        # Compute the offset in mm for y 
        yref_mm = np.diagonal(x_powers @ self.ymap_coeff @ y_powers.T)
        
        # now we get the y displacement (dispersion by wavelength)

        wavelength = np.atleast_1d(lam)
        # create power matrices, note that we still need x and y in fpa degrees 
        wl_powers = wavelength[:, np.newaxis] ** np.arange(6)  
        ref_powers = self.wl_ref ** np.arange(6) 
        x_pow = xfpa_deg[:, np.newaxis, np.newaxis, np.newaxis] ** np.arange(6)[np.newaxis, np.newaxis, :, np.newaxis]
        y_pow = yfpa_deg[:, np.newaxis, np.newaxis, np.newaxis] ** np.arange(6)[np.newaxis, np.newaxis, np.newaxis, :]
        ids_at_pos = np.sum(self.ids_coeff[np.newaxis, :, :, :] * x_pow * y_pow, axis=(2, 3))
        delta_y_mm = wl_powers @ ids_at_pos.T - (ref_powers @ ids_at_pos.T)
        delta_y_mm = np.diagonal(delta_y_mm)

        # now we get the x displacement (curvature)
        crv_at_pos = np.sum(self.crv_coeff[np.newaxis, :, :, :] * x_pow * y_pow, axis=(2, 3))
        dy_powers = delta_y_mm[:, np.newaxis] ** np.arange(6)
        delta_x_mm = dy_powers @ crv_at_pos.T
        delta_x_mm = np.diagonal(delta_x_mm)

        # now combine them and then convert to pixels and set the photon array coords to them
        xmpa_mm = xref_mm + delta_x_mm
        ympa_mm = yref_mm + delta_y_mm

        x_pix, y_pix = self.detector_coords.convert_mpa_to_sca(xmpa=xmpa_mm, ympa=ympa_mm, sca=self.sca)

        return x_pix, y_pix
        
    def applyTo(self, photon_array, local_wcs=None, rng=None):
        """Apply the grism dispersion to the photos
    
        Parameters:
            photon_array:   A `PhotonArray` to apply the operator to.
            local_wcs:      A `LocalWCS` instance defining the local WCS for the current photon
                            bundle in case the operator needs this information.  [default: None]
            rng:            A random number generator is not used.
        """
        #photon array has .x, .y, .wavelength, .coord, .time, ...
        if not photon_array.hasAllocatedWavelengths():
            raise GalSimError("Grism requires that wavelengths be set")
        
        # wavelength is in nm. Roman slitless thinks in microns.
        # http://galsim-developers.github.io/GalSim/_build/html/photon_array.html#galsim.PhotonArray
        w = photon_array.wavelength/1000.

        x_pix, y_pix = self.snpit_disperser.disperse(photon_array.x, photon_array.y, w, self.sca, self.order, pairwise=True)
        
        photon_array.x = x_pix
        photon_array.y = y_pix

    
    def __repr__(self):

        # )
        # s += ")"
        return f"galsim.GrismV()"

class WFSSSDisperserBuilder(PhotonOpBuilder):
    """Build a Grism op 
    """
    # This one needs special handling for obj_coord
    def buildPhotonOp(self, config, base, logger):
        req, opt, single, takes_rng = get_cls_params(WFSSSDisperser)
        kwargs, safe = GetAllParams(config, base, req, opt, single)
        #if 'sky_pos' in base:
        #    kwargs['obj_coord'] = base['sky_pos']
        return WFSSSDisperser(**kwargs)

RegisterPhotonOpType('WFSSSDisperser', WFSSSDisperserBuilder())