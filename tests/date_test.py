import yaml
import galsim
import galsim.config
import galsim.roman as roman
from astropy.time import Time

with open("config/was.yaml", "r") as f:
    base = yaml.safe_load(f)

config = base["image"]
base.setdefault("file_num", 0)
base.setdefault("image_num", 0)
base.setdefault("obj_num", 0)

galsim.config.ProcessInput(base)
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

params = galsim.config.GetAllParams(
    config, base, req=req, opt=opt, ignore=extra_ignore
)[0]
mjd = params["mjd"]
date = Time(mjd, format="mjd").to_datetime()

wcs_out = galsim.config.BuildWCS(config, 'wcs', base)
wcs = wcs_out[0] if isinstance(wcs_out, tuple) else wcs_out
xsize = config.get("xsize", config.get("size", roman.n_pix))
ysize = config.get("ysize", config.get("size", roman.n_pix))

# Dummy image just to get true_center in pixel coordinates
im = galsim.ImageF(xsize, ysize, wcs=wcs)
world_pos = wcs.toWorld(im.true_center)
sky_level = roman.getSkyLevel(base['bandpass'], world_pos=world_pos)
print("before sky_level =", sky_level)

sky_level = roman.getSkyLevel(base['bandpass'], world_pos=world_pos, date=date)
print("after sky_level =", sky_level)
