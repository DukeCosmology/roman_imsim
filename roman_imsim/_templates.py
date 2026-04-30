import os

import galsim

from ._meta_data import config_dir

galsim.config.RegisterTemplate("roman_imsim_default", os.path.join(config_dir, "default.yaml"))
