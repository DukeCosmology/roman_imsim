import romanisim.models as models
from galsim.config import BandpassBuilder, GetAllParams, RegisterBandpassType


class RomanBandpassBuilder(BandpassBuilder):
    """A class for loading a Bandpass from a file

    FileBandpass expected the following parameter:

        name (str)          The name of the Roman filter to get. (required)
    """

    def buildBandpass(self, config, base, logger):
        """Build the Bandpass based on the specifications in the config dict.

        Parameters:
            config:     The configuration dict for the bandpass type.
            base:       The base configuration dict.
            logger:     If provided, a logger for logging debug statements.

        Returns:
            the constructed Bandpass object.
        """
        req = {"name": str}
        opt = {"SCA": int}
        kwargs, safe = GetAllParams(config, base, req=req, opt=opt)

        name = kwargs["name"]
        if "SCA" in kwargs:
            bandpass = models.bandpass.getBandpasses(red_limit=2000, sca=kwargs["SCA"])[
                name
            ]
        else:
            bandpass = models.bandpass.getBandpasses(red_limit=2000)[name]

        return bandpass, safe


RegisterBandpassType("RomanBandpassTrimmed", RomanBandpassBuilder())
