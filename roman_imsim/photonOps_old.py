_w1 = 0.17519
_w2 = 0.53146
_w3 = 0.29335
_s  = 0.3279
_s1 = 0.4522*_s
_s2 = 0.8050*_s
_s3 = 1.4329*_s

import numpy as np
from galsim import PhotonOp,UniformDeviate,GaussianDeviate
from galsim.config import PhotonOpBuilder,RegisterPhotonOpType,get_cls_params,GetAllParams,GetRNG
import galsim

class ChargeDiff(PhotonOp):
    """A photon operator that applies the effect of charge diffusion via a probablistic model limit.
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

        self.ud   = UniformDeviate(rng)
        self.gd1  = GaussianDeviate(rng, sigma=_s1)
        self.gd2  = GaussianDeviate(rng, sigma=_s2)
        self.gd3  = GaussianDeviate(rng, sigma=_s3)

        # Choose which weighted Gausian to use in sech model approximation
        u  = np.empty(len(photon_array.x))
        self.ud.generate(u)

        # Selects appropriate fraction of photons corresponding to the first gaussian in the sech model
        mask = u<_w1
        dx = np.empty(np.sum(mask))
        dy = np.empty(np.sum(mask))
        # Generate and apply the 2D gaussian shifts corresponding to the first gaussian
        self.gd1.generate(dx)
        self.gd1.generate(dy)
        photon_array.x[mask] += dx
        photon_array.y[mask] += dy

        # Selects appropriate fraction of photons corresponding to the second gaussian in the sech model
        mask = (u>=_w1)&(u<=(1.-_w3))
        dx = np.empty(np.sum(mask))
        dy = np.empty(np.sum(mask))
        # Generate and apply the 2D gaussian shifts corresponding to the second gaussian
        self.gd2.generate(dx)
        self.gd2.generate(dy)
        photon_array.x[mask] += dx
        photon_array.y[mask] += dy

        # Selects appropriate fraction of photons corresponding to the third gaussian in the sech model 
        mask = u>(1.-_w3)
        dx = np.empty(np.sum(mask))
        dy = np.empty(np.sum(mask))
        # Generate and apply the 2D gaussian shifts corresponding to the second gaussian
        self.gd3.generate(dx)
        self.gd3.generate(dy)
        photon_array.x[mask] += dx
        photon_array.y[mask] += dy


class ChargeDiffBuilder(PhotonOpBuilder):
    """Build ChargeDiff photonOp
    """
    def buildPhotonOp(self, config, base, logger):
        req, opt, single, takes_rng = get_cls_params(ChargeDiff)
        kwargs, safe = GetAllParams(config, base, req, opt, single)
        rng = GetRNG(config, base, logger, "Roman_stamp")
        kwargs['rng'] = rng
        return ChargeDiff(**kwargs)

RegisterPhotonOpType('ChargeDiff', ChargeDiffBuilder())


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


import yaml
import numpy as np
from .optical_model_utils import RomanDetectorCoordinates

class GrismNV(PhotonOp):
    """A photon operator that applies the dispersion effects of the
    Roman Grism (nonvector form).
    
    The photons will need to have wavelengths defined in order to work.
        
    Parameters
        base_wavelength:    Wavelength (in nm) represented by the fiducial photon positions
    """
    # what parameters are tunable
    # _req_params = {"base_wavelength": float, "barycenter": list}
    # _opt_params = {"resolution": list}

    
    def __init__(self, config=None):
        if config is None:
            self.config = 'Roman_grism_OpticalModel_v0.8.yaml'
        else:
            self.config = config
        self.order = '1'
        self.sca = 16
        # self.base_wavelength = base_wavelength
        # self.resolution = np.array(resolution)
        with open(self.config) as f:
            data = yaml.safe_load(f)
        for order_key, order_data in data["optical_model"]["orders"].items():
            if order_key == self.order:
                order_dat = order_data
                break

        self.xmap_coeff = np.array(order_dat['xmap_ij_coeff'])
        self.ymap_coeff = np.array(order_dat['ymap_ij_coeff'])
        self.crv_coeff = np.array(order_dat['crv_ijk_coeff'])
        self.ids_coeff = np.array(order_dat['ids_ijk_coeff'])
        self.wl_min = 0.9 
        self.wl_max = 2.0
        self.wl_ref = 1.55

        # initialize RomanDetectorCoordinates
        # need these for some of the functions called
        self.detector_coords = RomanDetectorCoordinates(
            naxis1=data['detector_model'].get('naxis1', 4096),
            naxis2=data['detector_model'].get('naxis2', 4096),
            crpix1=data['detector_model'].get('crpix1', 2048),
            crpix2=data['detector_model'].get('crpix2', 2048),
            pos_angle_detector=data['detector_model'].get('pos_angle_detector', 0.0),
            pixel_scale=data['detector_model'].get('pixel_scale', 0.11),
            plate_scale=data['detector_model'].get('plate_scale', 100.0),
            xy_centers=data['detector_model'].get('xy_centers', {})
        )

    
    def _disperse(self, x0, y0, lam, sca, order='1'):
        # convert to FPA degrees
        xfpa_deg, yfpa_deg = self.detector_coords.convert_sca_to_fpa(
            photon_array.x, photon_array.y, sca=self.sca
        )
        # get the number of photons and the starting stuff
        nphot = len(wavelength)

        xref_mm = np.zeros(nphot)
        yref_mm = np.zeros(nphot)
        delta_y_mm = np.zeros(nphot)
        delta_x_mm = np.zeros(nphot)

        # loop through the photons and apply the IDS matrices
        for p in range(nphot):

            x = xfpa_deg[p]
            y = yfpa_deg[p]
            wl = wavelength[p]

            xr = 0.0
            yr = 0.0

            for i in range(6):
                xi = x ** i
                for j in range(6):
                    yj = y ** j
                    xr += self.xmap_coeff[i, j] * xi * yj
                    yr += self.ymap_coeff[i, j] * xi * yj

            xref_mm[p] = xr
            yref_mm[p] = yr


            # compute position-dependent coeffs C_i(x,y)
            C = np.zeros(6)

            for i in range(6):
                ci = 0.0
                for j in range(6):
                    xj = x ** j
                    for k in range(6):
                        yk = y ** k
                        ci += self.ids_coeff[i, j, k] * xj * yk
                C[i] = ci

            # apply wavelength polynomial (relative to wl_ref)
            dy = 0.0
            for i in range(6):
                dy += C[i] * (wl ** i - self.wl_ref ** i)

            delta_y_mm[p] = dy


            # curvature (x displacement as function of delta_y)

            D = np.zeros(6)

            for i in range(6):
                di = 0.0
                for j in range(6):
                    xj = x ** j
                    for k in range(6):
                        yk = y ** k
                        di += self.crv_coeff[i, j, k] * xj * yk
                D[i] = di

            dx = 0.0
            for i in range(6):
                dx += D[i] * (dy ** i)

            delta_x_mm[p] = dx

            xmpa_mm = xref_mm + delta_x_mm
            ympa_mm = yref_mm + delta_y_mm
            # go back to pixels

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
        wavelength = photon_array.wavelength/1000.


        # add to the photon array's positions
        photon_array.x = x_pix
        photon_array.y = y_pix

    
    def __repr__(self):
        # )
        # s += ")"
        s = "galsim.GrismNV()"
        return s

class GrismNVBuilder(PhotonOpBuilder):
    """Build a GrismNV
    """
    # This one needs special handling for obj_coord
    def buildPhotonOp(self, config, base, logger):
        req, opt, single, takes_rng = get_cls_params(GrismNV)
        kwargs, safe = GetAllParams(config, base, req, opt, single)
        #if 'sky_pos' in base:
        #    kwargs['obj_coord'] = base['sky_pos']
        return GrismNV(**kwargs)

RegisterPhotonOpType('GrismNV', GrismNVBuilder())

class GrismV(PhotonOp):
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
            self.config = "Roman_grism_OpticalModel_v0.8.yaml"
        else:
            self.config = config
        self.order = '1'
        self.sca = 16
        # self.base_wavelength = base_wavelength
        # self.resolution = np.array(resolution)
        with open(self.config) as f:
            data = yaml.safe_load(f)
        for order_key, order_data in data["optical_model"]["orders"].items():
            if order_key == self.order:
                order_dat = order_data
                break

        self.xmap_coeff = np.array(order_dat['xmap_ij_coeff'])
        self.ymap_coeff = np.array(order_dat['ymap_ij_coeff'])
        self.crv_coeff = np.array(order_dat['crv_ijk_coeff'])
        self.ids_coeff = np.array(order_dat['ids_ijk_coeff'])
        self.wl_min = 0.9 
        self.wl_max = 2.0
        self.wl_ref = 1.55
        # initialize RomanDetectorCoordinates
        # need these for some of the functions called
        self.detector_coords = RomanDetectorCoordinates(
            naxis1=data['detector_model'].get('naxis1', 4096),  # Adjust these defaults as needed
            naxis2=data['detector_model'].get('naxis2', 4096),
            crpix1=data['detector_model'].get('crpix1', 2048),
            crpix2=data['detector_model'].get('crpix2', 2048),
            pos_angle_detector=data['detector_model'].get('pos_angle_detector', 0.0),
            pixel_scale=data['detector_model'].get('pixel_scale', 0.11),
            plate_scale=data['detector_model'].get('plate_scale', 100.0),
            xy_centers=data['detector_model'].get('xy_centers', {})  # Dictionary of SCA centers
        )

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

        x_pix, y_pix = self._disperse(photon_array.x, photon_array.y, w, self.sca, self.order)
        
        photon_array.x = x_pix
        photon_array.y = y_pix

    
    def __repr__(self):

        # )
        # s += ")"
        return f"galsim.GrismV()"

class GrismVBuilder(PhotonOpBuilder):
    """Build a Grism op 
    """
    # This one needs special handling for obj_coord
    def buildPhotonOp(self, config, base, logger):
        req, opt, single, takes_rng = get_cls_params(GrismV)
        kwargs, safe = GetAllParams(config, base, req, opt, single)
        #if 'sky_pos' in base:
        #    kwargs['obj_coord'] = base['sky_pos']
        return GrismV(**kwargs)

RegisterPhotonOpType('GrismV', GrismVBuilder())
