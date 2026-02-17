import sys
import yaml
import galsim
import galsim.config
import galsim.roman as roman
import logging
import numpy as np
from astropy.time import Time

import roman_imsim
import roman_imsim.obseq


# --- read yaml ---
with open("config/was.yaml", "r") as f:
    base = yaml.safe_load(f)

config = base["image"]   # for an ImageBuilder-style config parse

# --- minimal runtime keys (often harmless, sometimes needed) ---
base.setdefault("file_num", 0)
base.setdefault("image_num", 0)
base.setdefault("obj_num", 0)

# --- logger ---
logger = logging.getLogger("date_test")
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

# Sanity check: do we even have the input block?
#print("input keys:", list(base.get("input", {}).keys()))
#print("obseq_data:", base.get("input", {}).get("obseq_data"))

# --- THIS is the missing step ---
galsim.config.ProcessInput(base, logger=logger)  # builds base['_input_objs'] :contentReference[oaicite:4]{index=4}

req = {"SCA": int, "filter": str, "mjd": float, "exptime": float}

opt = {
    "draw_method": str,
    "use_fft_bright": bool,
    "stray_light": bool,
    "thermal_background": bool,
    "reciprocity_failure": bool,
    "dark_current": bool,
    "nonlinearity": bool,
    "ipc": bool,
    "read_noise": bool,
    "sky_subtract": bool,
    "ignore_noise": bool,
}

extra_ignore = [
    "image_pos",
    "world_pos",
    "stamp_size",
    "stamp_xsize",
    "stamp_ysize",
    "nobjects",
    "wcs",
    "bandpass",
    "random_seed"
]

# Now your GetAllParams should be able to resolve mjd via ObSeqData -> GetInputObj
params = galsim.config.GetAllParams(
    config, base, req=req, opt=opt, ignore=extra_ignore
)[0]
mjd = params["mjd"]
#mjd = 60300
date = Time(mjd, format="mjd").to_datetime()
#print (mjd)
#print ("DATE!! ", date)

wcs_out = galsim.config.BuildWCS(config, 'wcs', base, logger=logger)
wcs = wcs_out[0] if isinstance(wcs_out, tuple) else wcs_out
xsize = config.get("xsize", config.get("size", roman.n_pix))
ysize = config.get("ysize", config.get("size", roman.n_pix))

# Dummy image just to get true_center in pixel coordinates
im = galsim.ImageF(xsize, ysize, wcs=wcs)

world_pos = wcs.toWorld(im.true_center)
sky_level = roman.getSkyLevel(base['bandpass'], world_pos=world_pos) #, date=date)
print("before sky_level =", sky_level)

sky_level = roman.getSkyLevel(base['bandpass'], world_pos=world_pos, date=date)
print("after sky_level =", sky_level)



