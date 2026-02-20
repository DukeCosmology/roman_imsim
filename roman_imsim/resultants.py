import galsim
import galsim.config
import galsim.roman as roman
import yaml
from galsim.config import InputLoader, RegisterInputType, RegisterValueType


class ResultantDataLoader(object):
    """Read the resultant information from the resultant strategy."""

    _req_params = {
        "file_name": str,
        "strategy_name": str,
    }

    def __init__(self, file_name, strategy_name, logger=None):
        self.logger = galsim.config.LoggerWrapper(logger)
        self.file_name = file_name
        self.strategy_name = strategy_name
        self.data = {}

        # try:
        self.read_resultants()
        # except:
        #     # Read visit info from the config file.
        #     self.logger.warning('Reading visit info from config file.')

    def read_resultants(self):
        """Load the YAML file and get the requested strategy."""
        self.logger.info("Reading resultants from YAML file: %s", self.file_name)
        try:
            with open(self.file_name, "r") as f:
                all_strategies = yaml.safe_load(f)
        except Exception as e:
            raise IOError(f"Could not read YAML file '{self.file_name}': {e}")

        if self.strategy_name not in all_strategies:
            raise ValueError(f"Strategy '{self.strategy_name}' not found in YAML file.")

        strategy = all_strategies[self.strategy_name]
        if not isinstance(strategy, list):
            raise ValueError(f"Invalid strategy format for '{self.strategy_name}': must be a list of lists.")

        self.data["strategy"] = strategy

    def resultants_to_dt(self, config, base):
        """Compute dt from list-of-lists."""
        strategy = self.data["strategy"]
        if len(strategy) < 2:
            raise ValueError("Need at least two resultants to compute dt.")

        avg_last = sum(strategy[-1]) / len(strategy[-1])
        avg_second = sum(strategy[0]) / len(strategy[0])

        if "exptime" in config:
            exptime = galsim.config.ParseValue(config, "exptime", base, float)[0]
        else:
            exptime = roman.exptime

        dt = exptime / (avg_last - avg_second)
        return dt

    def get(self, field, default=None):
        if field not in self.data and default is None:
            raise KeyError(f"Field '{field}' not found in data.")
        return self.data.get(field, default)


def ResultantData(config, base, value_type):
    """Returns the resultant dt data."""
    rdata = galsim.config.GetInputObj("resultant_data", config, base, "ResultantDataLoader")
    req = {"field": str}
    kwargs, safe = galsim.config.GetAllParams(config, base, req=req)
    field = kwargs["field"]

    if field == "dt":
        val = rdata.resultants_to_dt(config, base)
    else:
        val = rdata.get(field)

    return value_type(val), safe


RegisterInputType("resultant_data", InputLoader(ResultantDataLoader, file_scope=True, takes_logger=True))
RegisterValueType("ResultantData", ResultantData, [float, list], input_type="resultant_data")
