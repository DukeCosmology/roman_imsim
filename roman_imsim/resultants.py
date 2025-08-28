import fitsio as fio
import galsim
import galsim.config
from galsim.config import InputLoader, RegisterInputType, RegisterValueType, RegisterObjectType
import galsim.roman as roman

class ResultantDataLoader(object):
    """Read the resultant information from the resultant strategy."""

    _req_params = {"file_name": str, "strategy": str,}

    def __init__(self, file_name, strategy, logger=None):
        self.logger = galsim.config.LoggerWrapper(logger)
        self.file_name = file_name
        self.strategy = strategy


        # try:
        self.read_resultants()
        # except:
        #     # Read visit info from the config file.
        #     self.logger.warning('Reading visit info from config file.')

    def read_resultants(self):
        """Read resultant info from the resultants file."""
        if self.file_name is None:
            raise ValueError("No resultants filename provided, trying to build from config information.")
        if self.strategy is None:
            raise ValueError("The strategy must be set when reading resultant strategy info from a resultants file.")

        self.logger.warning("Reading info from resultants file %s for strategy %s", self.file_name, self.strategy)

        data = fio.FITS(self.file_name)[-1][self.strategy]

        self.data = {}
        self.data["strategy"] = data["strategy"]
        self.data["dt"] = self.resultants_to_dt()

    def resultants_to_dt(self,config,base):
        if "exptime" in config:
            exptime = galsim.config.ParseValue(config, "exptime", base, float)[0]
        else:
            exptime = roman.exptime
        dt = exptime/((sum(self.strategy[-1])/len(self.strategy[-1]))-(sum(self.strategy[1])/len(self.strategy[1])))
        return dt
    
    def get(self, field, default=None):
        if field not in self.data and default is None:
            raise KeyError("ResultantData field %s not present in data" % field)
        return self.data.get(field, default)
    
def ResultantData(config, base, value_type):
    """Returns the resultant dt data."""
    rdata = galsim.config.GetInputObj("resultant_data", config, base, "ResultantDataLoader")
    req = {"field": str}
    kwargs, safe = galsim.config.GetAllParams(config, base, req=req)
    field = kwargs["field"]

    val = value_type(rdata.get(field))
    return val, safe


RegisterInputType("resultant_data", InputLoader(ResultantDataLoader, file_scope=True, takes_logger=True))
RegisterValueType("ResultantData", ResultantData, [float,list], input_type="resultant_data")

