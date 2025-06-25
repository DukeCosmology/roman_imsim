from itertools import product
import numpy as np

import galsim
import galsim.config
import galsim.roman as roman

from skycat import SkyCatalogInterface

class roman_utils(object):
    """
    Class to contain a variety of helper routines to work with the simulation data.
    """

    def __init__(self, config_file, visit=None, sca=None, image_name=None, setup_skycat=False):
        """
        Setup information about a simulated Roman image.
        Parameters:
            config_file: the GalSim config file that produced the simulation
            visit: the visit (observation sequence) number of the pointing
            sca: the SCA number
            image_name: the filename of the image (can be used instead of visit, sca)
            setup_skycat: setup the skycatalog information to have access to
        """
        config = galsim.config.ReadConfig(config_file)[0]

        self.visit, self.sca = self.check_input(visit, sca, image_name)

        if not setup_skycat:
            del config["input"]["sky_catalog"]
        config["input"]["obseq_data"]["visit"] = self.visit
        config["image"]["SCA"] = self.sca
        galsim.config.ProcessInput(config)
        if setup_skycat:
            self.skycat = galsim.config.GetInputObj(
                "sky_catalog", config["input"]["sky_catalog"], config, "sky_catalog"
            )
        self.PSF = galsim.config.GetInputObj("roman_psf", config["input"]["roman_psf"], config, "roman_psf")
        self.wcs = galsim.config.BuildWCS(config["image"], "wcs", config)
        self.bpass = galsim.config.BuildBandpass(config["image"], "bandpass", config, None)[0]
        self.photon_ops = galsim.config.BuildPhotonOps(config["stamp"], "photon_ops", config, None)
        self.rng = galsim.config.GetRNG(config, config["image"], None, "roman_sca")

    def check_input(self, visit, sca, image_name):
        if image_name is not None:
            print("Inferring visit and sca from image_name.")
            start = 21
            end = -5
            if "simple_model" in image_name:
                start = 28
            if "gz" in image_name:
                end = -8
            tmp = np.array(image_name[start:end].split("_")).astype(int)
            return tmp[0], tmp[1]
        if (visit is None) | (sca is None):
            raise ValueError("Insufficient information to construct visit info - all inputs are None.")
        return visit, sca

    def getPSF(self, x=None, y=None, pupil_bin=8):
        """
        Return Roman PSF for some image position.
        Parameters:
            x: x-position in SCA
            y: y-position in SCA
            pupil_bin: pupil image binning factor
        Returns:
            the chromatic GalSim PSF model object (does not include additional effects like charge diffusion!)
        """
        if pupil_bin != 8:
            if (x is not None) | (y is not None):
                raise ValueError(
                    "x,y position for pupil_bin values other than 8 not supported. Using SCA center."
                )
            return self.PSF.getPSF(pupil_bin, galsim.PositionD(roman.n_pix / 2, roman.n_pix / 2))
        if (x is None) | (y is None):
            return self.PSF.getPSF(8, galsim.PositionD(roman.n_pix / 2, roman.n_pix / 2))
        return self.PSF.getPSF(8, galsim.PositionD(x, y))

    def getWCS(self):
        """
        Return Roman WCS for image
        """
        return self.wcs

    def getLocalWCS(self, x, y):
        """
        Return Roman WCS for image
        """
        return self.wcs.local(galsim.PositionD(x, y))

    def getBandpass(self):
        """
        Return Roman bandpass for image
        """
        return self.bpass

    def getPSF_Image(
        self,
        stamp_size,
        x=None,
        y=None,
        pupil_bin=8,
        sed=None,
        oversampling_factor=1,
        include_photonOps=False,
        n_phot=1e6,
        method="no_pixel",
        include_pixel=False,
    ):
        """
        Return a Roman PSF image for some image position
        Parameters:
            stamp_size: size of output PSF model stamp in native roman pixel_scale (oversampling_factor=1)
            x: x-position in SCA
            y: y-position in SCA
            pupil_bin: pupil image binning factor
            sed: SED to be used to draw the PSF - default is a flat SED.
            oversampling_factor: factor by which to oversample native roman pixel scale
            include_photonOps: include additional contributions from other photon operators in effective psf
                               image
            n_phot: Number of photons to generate PSF using photon shooting method.
                    Relevant only if include_photonOps is True (default: 1e6)
            method: Method used to generate the PSF image.
                    Relevant only if include_photoOps is False (default: 'no_pixel')
            include_pixel: convolve output to have the Roman pixel response (default: False)
        Returns:
            the PSF GalSim image object (use image.array to get a numpy array representation)
        """
        if sed is None:
            sed = galsim.SED(
                galsim.LookupTable([400, 2600], [1, 1], interpolant="linear"),
                wave_type="nm",
                flux_type="fphotons",
            )
        point = galsim.DeltaFunction(flux=1) * sed
        point = point.withFlux(1, self.bpass)
        local_wcs = self.getLocalWCS(x, y)
        wcs = galsim.JacobianWCS(
            dudx=local_wcs.dudx / oversampling_factor,
            dudy=local_wcs.dudy / oversampling_factor,
            dvdx=local_wcs.dvdx / oversampling_factor,
            dvdy=local_wcs.dvdy / oversampling_factor,
        )
        stamp = galsim.Image(stamp_size * oversampling_factor, stamp_size * oversampling_factor, wcs=wcs)
        if not include_photonOps:
            psf = galsim.Convolve(point, self.getPSF(x, y, pupil_bin))
            return psf.drawImage(self.bpass, image=stamp, wcs=wcs, method=method)
        photon_ops = [self.getPSF(x, y, pupil_bin)] + self.photon_ops
        if include_pixel and oversampling_factor > 1:
            scale = galsim.roman.pixel_scale
            # An array (comb) of Dirac Delta functions.
            comb = galsim.Add(
                [
                    galsim.DeltaFunction(1.0).shift(
                        dx=i * scale / oversampling_factor,
                        dy=j * scale / oversampling_factor,
                    )
                    for i, j in product(
                        np.arange((1 - oversampling_factor) / 2, (1 + oversampling_factor) / 2),
                        np.arange((1 - oversampling_factor) / 2, (1 + oversampling_factor) / 2),
                    )
                ]
            )
            point = galsim.Convolve(point, comb)
        return point.drawImage(
            self.bpass,
            method="phot",
            rng=self.rng,
            maxN=int(1e6),
            n_photons=int(n_phot),
            image=stamp,
            photon_ops=photon_ops,
            poisson_flux=False,
        )

class transient_utils(object):
    """
    Class to assist with retrieving transient information.
    """

    def __init__(
        self,
        file_name,
        obj_id
    ):

        skycat_int = SkyCatalogInterface( 
                                        file_name,
                                        exptime = 30,  # Arbitrary, re-defines this later
                                        )

        return skycat_int