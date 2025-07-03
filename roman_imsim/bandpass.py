import galsim
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
        req = {"name": int}
        kwargs, safe = GetAllParams(config, base, req=req)

        name = kwargs["name"]
        if int(name) > 747:
            return None

        # Define wavelength range
        def dl(l_, R):
            return l_ / R

        # Define next lambda center
        def lc(ll, R):
            return ll / (1 - 1 / 2.0 / R)

        # Define next lambda right
        def lr_(ll, R):
            return lc(ll, R) + dl(ll, R) / 2.0

        # Wavelength range in microns
        lambda_min = 0.2 * 10000
        # lambda_max = 2.5 * 10000
        R = 300  # spectral resolution

        def tophat(wl):
            return 1.0

        ll = lambda_min
        lr = lr_(ll, R)
        if int(name) != 0:
            for i in range(int(name)):
                ll = lr
                lr = lr_(ll, R)

        # Set up a dictionary.
        bandpass_dict = {}

        bp = galsim.Bandpass(tophat, blue_limit=ll, red_limit=lr, wave_type="Ang")

        # Add it to the dictionary.
        bp.name = name
        bandpass_dict[bp.name] = bp

        return bp, safe


RegisterBandpassType("RomanBandpassTrimmed", RomanBandpassBuilder())
