import galsim
import galsim.config
import romanisim.models as models

from .ramp import validate_read_pattern
import yaml
import math
from galsim.config import InputLoader, RegisterInputType, RegisterValueType


class ResultantDataLoader(object):
    """Read the resultant information from the resultant strategy."""

    _opt_params = {
        "file_name": str,
        "strategy_name": str,
        "read_pattern": list,
        "ma_table_number": int,
    }

    def __init__(self, file_name=None, strategy_name=None, read_pattern=None, ma_table_number=None, logger=None):
        self.logger = galsim.config.LoggerWrapper(logger)
        self.file_name = file_name
        self.strategy_name = strategy_name
        self.data = {}

        choices = sum([file_name is not None, read_pattern is not None, ma_table_number is not None])
        if choices != 1:
            raise ValueError('Specify exactly one of file_name, read_pattern, or ma_table_number')
        if file_name is not None:
            if strategy_name is None:
                raise ValueError('strategy_name is required with file_name')
            self.read_resultants()
        else:
            if strategy_name is not None:
                raise ValueError('strategy_name is only valid with file_name')
            if ma_table_number is not None:
                try:
                    read_pattern = models.parameters.read_pattern[ma_table_number]
                except KeyError as e:
                    raise ValueError(f'Unknown MA table {ma_table_number}') from e
                self.data['ma_table_number'] = ma_table_number
            self.data['strategy'] = validate_read_pattern(read_pattern)
        self.data['frame_time'] = models.parameters.read_time
        self.data['exptime'] = models.parameters.read_time * self.data['strategy'][-1][-1]

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

        self.data["strategy"] = validate_read_pattern(strategy)

    def resultants_to_dt(self, config, base):
        """Compute dt from list-of-lists."""
        strategy = self.data["strategy"]
        image_config = base.get("image", config)
        if "exptime" in image_config:
            exptime = galsim.config.ParseValue(image_config, "exptime", base, float)[0]
        else:
            exptime = models.parameters.exptime
        if not math.isfinite(exptime) or not exptime > 0:
            raise ValueError("Exposure time must be positive")
        return exptime / strategy[-1][-1]


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
        safe = False  # cadence depends on this exposure
    else:
        val = rdata.get(field)

    return value_type(val), safe


RegisterInputType("resultant_data", InputLoader(ResultantDataLoader, file_scope=True, takes_logger=True))
RegisterValueType("ResultantData", ResultantData, [float, list], input_type="resultant_data")
