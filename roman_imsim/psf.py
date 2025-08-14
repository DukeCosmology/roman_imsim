import galsim
import galsim.config
import galsim.roman as roman
from galsim.config import InputLoader, RegisterInputType


class RomanPSF(object):
    """Class building needed Roman PSFs."""

    def __init__(
        self,
        n_waves=None,
        extra_aberrations=None,
    ):

        self._n_waves = n_waves
        self._extra_aberrations = extra_aberrations

    def _parse_pupil_bin(self, pupil_bin):
        if pupil_bin == "achromatic":
            return 8
        else:
            return pupil_bin

    def _psf_call(self, SCA, bpass, SCA_pos, WCS, pupil_bin, n_waves, logger, extra_aberrations):

        if pupil_bin == 8:
            psf = roman.getPSF(
                SCA,
                bpass.name,
                SCA_pos=SCA_pos,
                wcs=WCS,
                pupil_bin=pupil_bin,
                n_waves=n_waves,
                logger=logger,
                # Don't set wavelength for this one.
                # We want this to be chromatic for photon shooting.
                # wavelength          = bpass.effective_wavelength,
                extra_aberrations=extra_aberrations,
            )
        else:
            psf = roman.getPSF(
                SCA,
                bpass.name,
                SCA_pos=SCA_pos,
                wcs=WCS,
                pupil_bin=self._parse_pupil_bin(pupil_bin),
                n_waves=n_waves,
                logger=logger,
                # Note: setting wavelength makes it achromatic.
                # We only use pupil_bin = 2,4 for FFT objects.
                wavelength=bpass.effective_wavelength,
                extra_aberrations=extra_aberrations,
            )
        if pupil_bin == 4:
            return psf.withGSParams(maximum_fft_size=16384, folding_threshold=1e-3)
        elif pupil_bin == 2:
            return psf.withGSParams(maximum_fft_size=16384, folding_threshold=1e-4)
        else:
            return psf.withGSParams(maximum_fft_size=16384)

    def initPSF(
        self,
        SCA=None,
        WCS=None,
        bandpass=None,
        image_xsize=None,
        image_ysize=None,
        logger=None,
    ):

        logger = galsim.config.LoggerWrapper(logger)

        self.SCA = SCA

        n_waves = self._n_waves
        if n_waves == -1:
            if bandpass.name == "W146":
                n_waves = 10
            else:
                n_waves = 5

        self._image_xsize = image_xsize
        if self._image_xsize is None:
            self._image_xsize = roman.n_pix
        self._image_ysize = image_ysize
        if self._image_ysize is None:
            self._image_ysize = roman.n_pix

        corners = [
            galsim.PositionD(1, 1),
            galsim.PositionD(1, self._image_ysize),
            galsim.PositionD(self._image_xsize, 1),
            galsim.PositionD(self._image_xsize, self._image_ysize),
        ]
        cc = galsim.PositionD(self._image_xsize / 2, self._image_ysize / 2)
        tags = ["ll", "lu", "ul", "uu"]
        self.PSF = {}
        pupil_bin = 8
        self.PSF[pupil_bin] = {}
        for tag, SCA_pos in tuple(zip(tags, corners)):
            self.PSF[pupil_bin][tag] = self._psf_call(
                SCA, bandpass, SCA_pos, WCS, pupil_bin, n_waves, logger, self._extra_aberrations
            )
        for pupil_bin in [4, 2, "achromatic"]:
            self.PSF[pupil_bin] = self._psf_call(
                SCA, bandpass, cc, WCS, pupil_bin, n_waves, logger, self._extra_aberrations
            )

    def getPSF(self, pupil_bin, pos):
        """
        Return a PSF to be convolved with sources.

        @param [in] what pupil binning to request.
        """

        # temporary
        # psf = self.PSF[pupil_bin]['ll']
        # if ((pos.x-roman.n_pix)**2+(pos.y-roman.n_pix)**2)<((pos.x-1)**2+(pos.y-1)**2):
        #     psf = self.PSF[pupil_bin]['uu']
        # if ((pos.x-1)**2+(pos.y-roman.n_pix)**2)<((pos.x-roman.n_pix)**2+(pos.y-roman.n_pix)**2):
        #     psf = self.PSF[pupil_bin]['lu']
        # if ((pos.x-roman.n_pix)**2+(pos.y-1)**2)<((pos.x-1)**2+(pos.y-roman.n_pix)**2):
        #     psf = self.PSF[pupil_bin]['ul']
        # if ((pos.x-roman.n_pix/2)**2+(pos.y-roman.n_pix/2)**2)<((pos.x-roman.n_pix)**2+(pos.y-1)**2):
        #     psf = self.PSF[pupil_bin]['cc']
        # return psf

        psf = self.PSF[pupil_bin]
        if pupil_bin != 8:
            return psf

        wll = (self._image_xsize - pos.x) * (self._image_ysize - pos.y)
        wlu = (self._image_xsize - pos.x) * (pos.y - 1)
        wul = (pos.x - 1) * (self._image_ysize - pos.y)
        wuu = (pos.x - 1) * (pos.y - 1)
        return (wll * psf["ll"] + wlu * psf["lu"] + wul * psf["ul"] + wuu * psf["uu"]) / (
            (self._image_xsize - 1) * (self._image_ysize - 1)
        )


class PSFLoader(InputLoader):
    """PSF loader."""

    def __init__(self):
        # Override some defaults in the base init.
        super().__init__(init_func=RomanPSF, takes_logger=True, use_proxy=False)

    def getKwargs(self, config, base, logger):
        logger.debug("Get kwargs for PSF")

        req = {}
        opt = {
            "n_waves": int,
        }
        ignore = ["extra_aberrations"]

        kwargs, safe = galsim.config.GetAllParams(config, base, req=req, opt=opt, ignore=ignore)

        kwargs["extra_aberrations"] = galsim.config.ParseAberrations(
            "extra_aberrations", config, base, "RomanPSF"
        )

        return kwargs, safe

    def setupImage(self, input_obj, config, base, logger):
        """ """

        bandpass = galsim.config.BuildBandpass(base["image"], "bandpass", base, logger)[0]

        input_obj.initPSF(
            SCA=base["SCA"],
            WCS=base["wcs"],
            bandpass=bandpass,
            image_xsize=base["image_xsize"],
            image_ysize=base["image_ysize"],
            logger=logger,
        )


# Register this as a valid type
RegisterInputType("roman_psf", PSFLoader())
# RegisterObjectType('roman_psf', BuildRomanPSF, input_type='romanpsf_loader')
