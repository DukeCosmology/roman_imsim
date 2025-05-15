import galsim.roman as roman
from galsim.config import BandpassBuilder, RegisterBandpassType, GetAllParams


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
        kwargs, safe = GetAllParams(config, base, req=req)

        name = kwargs["name"]
        bandpass = roman.getBandpasses(red_limit=2000)[name]

        return bandpass, safe


RegisterBandpassType("RomanBandpassTrimmed", RomanBandpassBuilder())
