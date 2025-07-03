import galsim
from galsim.celestial import CelestialCoord
from galsim.config import RegisterWCSType, WCSBuilder


class RomanWCS(WCSBuilder):

    def buildWCS(self, config, base, logger):

        req = {
            "ra": float,
            "dec": float,
        }
        opt = {}
        kwargs, safe = galsim.config.GetAllParams(config, base, req=req, opt=opt)
        pointing = CelestialCoord(ra=kwargs["ra"] * galsim.degrees, dec=kwargs["dec"] * galsim.degrees)
        wcs = galsim.TanWCS(
            affine=galsim.AffineTransform(dudx=0.003, dudy=0.0, dvdx=0.0, dvdy=0.003),
            world_origin=pointing,
        )

        return wcs


RegisterWCSType("RomanWCS2", RomanWCS())
