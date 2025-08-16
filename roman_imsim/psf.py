"""
Implementation of the RomanPSF builder

RomanPSF is registered as a new GSObject that can be used within the `psf` section of the config file.
It can be used in a few different ways depending if you want to draw the PSF at each locations from scratch or
if you want to use interpolation to speed things up.

Here is how the PSF Builder works:
----------------------------------

    RomanPSF
        |
        |------ No interpolation ---- Call the galsim.roman.getPSF function and parameters of this function
        |                             can be set from the config file
        |
        |
        |------ Interpolation ------- It requires the `interpolator` keyword in the config file and you need
                    |                 to specify the type of interpolator, at the moment there is only
                    |                 `RomanPSFInterpolator` available.
                    |
                    |------- Interpolator loader ---- To use an interpolator it needs to be initialized and
                                    |                 for that we need to declare it in the `input` section of
                                    |                 the config file as: `RomanPSFInterpolator`.
                                    |
                                    |----- Interpolation kind ---- For the `RomanPSFInterpolator` in the input
                                                                   you need to specify which `kind` of
                                                                   interpolation to use. At the moment only
                                                                   `corners` is availalbe.

Adding a new interpolation kind:
--------------------------------

You need to create it from the `PSFInterpolator` class and implement the `initPSF` and `getPSF` methods. Then
you just need to register it to make GalSim aware of it by doing:
```python
RegisterPSFInterpolatorType("new_interpolator", NewPSFInterpolator)
```
And it will be automatically available from the config file by doing:
```yaml
input:
  RomanPSFInterpolator:
    kind: new_interpolator
```
"""

import galsim
import galsim.roman as roman
from galsim.config import (
    InputLoader,
    RegisterInputType,
    RegisterInputConnectedType,
    RegisterValueType,
    RegisterObjectType,
)
from galsim.errors import GalSimConfigValueError

##########################
# PSF Interpolator Input #
##########################

valid_psf_interpolator_types = {}


class PSFInterpolator:
    """Base class for PSF interpolator"""

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
        raise NotImplementedError("initPSF must be implemented in subclasses")

    def getPSF(self, pupil_bin, pos):
        raise NotImplementedError("getPSF must be implemented in subclasses")


class CornerPSFInterpolator(PSFInterpolator):
    """Corner PSF interpolator

    This class allows to builds the PSF in the corner of the image and center. Then instead of using the
    "real" PSF variation of the PSF, it will be interpolated at the desired position. This saved a significant
    amount of computation time.

    Parameters
    ----------
    n_waves : int
        Number of wavelengths to use for the Chromatic PSF interpolation in GalSim.
    extra_aberrations : dict
        Additional aberrations to apply to the PSF. (Not supported)
    """

    def __init__(
        self,
        n_waves=None,
        extra_aberrations=None,
    ):

        self._n_waves = n_waves
        self._extra_aberrations = extra_aberrations

    def initPSF(
        self,
        SCA=None,
        WCS=None,
        bandpass=None,
        image_xsize=None,
        image_ysize=None,
        logger=None,
    ):
        """Initialize the PSF interpolator.

        This function sets up the PSF interpolator by building the PSF at the required positions for the
        interpolation.

        Parameters
        ----------
        SCA: int
            SCA for which we build the interpolated PSF
        WCS: galsim.WCS
            WCS to use for the PSF
        bandpass: galsim.Bandpass
            Bandpass filter to use for the PSF
        image_xsize: int
            Size of the image in the x direction
        image_ysize: int
            Size of the image in the y direction
        logger: logging.Logger
            Logger
        """

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

        Parameters
        ----------
        pupil_bin : int
            Pupil binning to use for the PSF
        pos : galsim.PositionD
            Image position at which to evaluate the PSF

        Return
        ------
        galsim.GSObject
            The PSF to be convolved with sources.
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


def RegisterPSFInterpolatorType(interp_type, builder, input_type=None):
    """Register a PSF interpolator type for use by the config apparatus.

    Parameters
    ----------
        interp_type : str
            The name of the type in the config dict.
        builder: PSFInterpolator
            A builder object to use for building the PSF interpolator.  It should
            be an instance of a subclass of PSFInterpolator.
        input_type: str or list of str
            If the PSF interpolator utilises an input object, give the key name of the
            input type here. (If it uses more than one, this may be a list.)
    """
    valid_psf_interpolator_types[interp_type] = builder
    RegisterInputConnectedType(input_type, interp_type)


RegisterPSFInterpolatorType("corners", CornerPSFInterpolator)


class PSFInterpolatorLoader(InputLoader):
    """PSF loader"""

    def getKwargs(self, config, base, logger):
        logger.debug("Get kwargs for PSF")

        req = {
            "kind": str,
        }
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
        """
        The PSF interpolator is initialized for each image.

        NOTE: maybe look into how to bypass the initialization in case the new image has the same properties
        as the existing one. (check on SCA, WCS, bandpass)
        """

        bandpass = galsim.config.BuildBandpass(base["image"], "bandpass", base, logger)[0]

        input_obj.initPSF(
            SCA=base["SCA"],
            WCS=base["wcs"],
            bandpass=bandpass,
            image_xsize=base["image_xsize"],
            image_ysize=base["image_ysize"],
            logger=logger,
        )


def PSFInterpolatorLoaderHelper(kind, **kwargs):
    """
    Helper function to load a specific kind of PSF interpolator.

    Parameters
    ----------
        kind : str
            The kind of PSF interpolator to load.
        **kwargs : dict
            Additional keyword arguments to pass to the interpolator.

    Returns
    -------
        PSFInterpolator
            The loaded PSF interpolator.
    """

    if kind not in valid_psf_interpolator_types:
        raise GalSimConfigValueError(
            "Invalid interpolator.kind", kind, list(valid_psf_interpolator_types.keys())
        )

    return valid_psf_interpolator_types[kind](**kwargs)


# Register this as a valid type
# NOTE: If you add a new PSF interpolator, you do NOT need to change this. This is the "magic" of the helper
#       function.
RegisterInputType(
    "RomanPSFInterpolator",
    PSFInterpolatorLoader(
        PSFInterpolatorLoaderHelper,
        takes_logger=True,
        use_proxy=False,
    ),
)


##########################
# PSF Interpolator Value #
##########################


def RomanPSFInterpolator(config, base, value_type):
    """
    Value type for the Roman PSF interpolator.

    This allows to specify the type of PSF interpolator to use when building the Roman PSF. It will directly
    return the initialized interpolator based on the `kind` of interpolator specified in the `input` section.
    """

    req = {
        "type": str,
    }

    params, safe = galsim.config.GetAllParams(config, base, req=req)

    interpolator = galsim.config.GetInputObj(params["type"], config, base, "RomanPSFInterpolator")

    if not isinstance(interpolator, value_type):
        raise TypeError(f"Invalid interpolator type. Got: {type(interpolator)} instead of {value_type}")

    return interpolator, safe


# NOTE: If you add a new PSF interpolator, you do NOT need to change this.
RegisterValueType(
    "RomanPSFInterpolator",
    RomanPSFInterpolator,
    [PSFInterpolator],
    input_type=[
        "PSFInterpolatorLoader",
    ],
)


####################
# Roman PSF Object #
####################


def BuildRomanPSF(config, base, ignore, gsparams, logger):
    """
    Roman PSF builder.

    This function builds the Roman PSF and will use an interpolator if one is provided in the config.
    """

    req = {}
    opt = {
        "pupil_bin": int,
        "n_waves": int,
        "wavelength": float,
        "interpolator": PSFInterpolator,
    }
    extra_ignore = [
        "extra_aberrations",
    ]

    params, safe = galsim.config.GetAllParams(config, base, req=req, opt=opt, ignore=ignore + extra_ignore)

    if "interpolator" in params:
        builder = params["interpolator"]
        psf = builder.getPSF(
            base["pupil_bin"],
            base["image_pos"],
        )
    else:
        psf = roman.getPSF(
            SCA=base["SCA"],
            bandpass=base["bandpass"].name,
            SCA_pos=base["image_pos"],
            pupil_bin=params.get("pupil_bin", base.get("pupil_bin", 4)),
            wcs=base["wcs"],
            n_waves=params.get("n_waves", None),
            extra_aberrations=None,
            wavelength=params.get("wavelength", None),
            gsparams=galsim.GSParams(**gsparams),
            logger=logger,
            high_accuracy=None,
            approximate_struts=None,
        )
    safe = False
    return psf, safe


RegisterObjectType("RomanPSF", BuildRomanPSF)
