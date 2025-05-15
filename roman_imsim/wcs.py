import galsim
import galsim.roman as roman
from astropy.time import Time
from galsim.angle import Angle
from galsim.celestial import CelestialCoord
from galsim.config import RegisterWCSType, WCSBuilder


class RomanWCS(WCSBuilder):

    def buildWCS(self, config, base, logger):

        req = {
            "SCA": int,
            "ra": Angle,
            "dec": Angle,
            "pa": Angle,
            "mjd": float,
        }
        opt = {"max_sun_angle": float, "force_cvz": bool}

        kwargs, safe = galsim.config.GetAllParams(config, base, req=req, opt=opt)
        if "max_sun_angle" in kwargs:
            roman.max_sun_angle = kwargs["max_sun_angle"]
            roman.roman_wcs.max_sun_angle = kwargs["max_sun_angle"]
        pointing = CelestialCoord(ra=kwargs["ra"], dec=kwargs["dec"])
        wcs = roman.getWCS(
            world_pos=pointing,
            PA=kwargs["pa"],
            date=Time(kwargs["mjd"], format="mjd").datetime,
            SCAs=kwargs["SCA"],
            PA_is_FPA=True,
        )[kwargs["SCA"]]
        return wcs


RegisterWCSType("RomanWCS", RomanWCS())
