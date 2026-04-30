import fitsio as fio
import galsim
import galsim.config
from astropy.time import Time
from galsim.angle import Angle
from galsim.config import InputLoader, RegisterInputType, RegisterValueType


class ObSeqDataLoader(object):
    """Read the exposure information from the observation sequence."""

    _req_params = {"file_name": str, "visit": int, "SCA": int}

    def __init__(self, file_name, visit, SCA, logger=None):
        self.logger = galsim.config.LoggerWrapper(logger)
        self.file_name = file_name
        self.visit = visit
        self.sca = SCA

        # try:
        self.read_obseq()
        # except:
        #     # Read visit info from the config file.
        #     self.logger.warning('Reading visit info from config file.')

    def read_obseq(self):
        """Read visit info from the obseq file."""
        if self.file_name is None:
            raise ValueError(
                "No obseq filename provided, trying to build from config information."
            )
        if self.visit is None:
            raise ValueError(
                "The visit must be set when reading visit info from an obseq file."
            )

        self.logger.warning(
            "Reading info from obseq file %s for visit %s", self.file_name, self.visit
        )

        ob = fio.FITS(self.file_name)[-1][self.visit]

        self.ob = {}
        self.ob["visit"] = self.visit
        self.ob["sca"] = self.sca
        self.ob["ra"] = ob["ra"] * galsim.degrees
        self.ob["dec"] = ob["dec"] * galsim.degrees
        self.ob["pa"] = ob["pa"] * galsim.degrees
        self.ob["date"] = Time(ob["date"], format="mjd").datetime
        self.ob["mjd"] = ob["date"]
        self.ob["filter"] = ob["filter"]
        self.ob["exptime"] = ob["exptime"]

    def get(self, field, default=None):
        if field not in self.ob and default is None:
            raise KeyError("OpsimData field %s not present in ob" % field)
        return self.ob.get(field, default)


def ObSeqData(config, base, value_type):
    """Returns the obseq data for a pointing."""
    pointing = galsim.config.GetInputObj("obseq_data", config, base, "OpSeqDataLoader")
    req = {"field": str}
    kwargs, safe = galsim.config.GetAllParams(config, base, req=req)
    field = kwargs["field"]

    val = value_type(pointing.get(field))
    return val, safe


RegisterInputType(
    "obseq_data", InputLoader(ObSeqDataLoader, file_scope=True, takes_logger=True)
)
RegisterValueType(
    "ObSeqData", ObSeqData, [float, int, str, Angle], input_type="obseq_data"
)
