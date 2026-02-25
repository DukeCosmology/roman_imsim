"""
Interface to obtain objects from skyCatalogs.
"""

import galsim
import galsim.roman as roman
import numpy as np
from galsim.config import (
    InputLoader,
    RegisterInputType,
    RegisterObjectType,
    RegisterValueType,
)


class SkyCatalogInterface:
    """Interface to skyCatalogs package."""

    _trivial_sed = galsim.SED(
        galsim.LookupTable([100, 2600], [1, 1], interpolant="linear"),
        wave_type="nm",
        flux_type="fphotons",
    )

    def __init__(
        self,
        file_name,
        exptime,
        wcs=None,
        mjd=None,
        bandpass=None,
        xsize=None,
        ysize=None,
        obj_types=None,
        edge_pix=100,
        max_flux=None,
        logger=None,
    ):
        """
        Parameters
        ----------
        file_name : str
            Name of skyCatalogs yaml config file.
        wcs : galsim.WCS
            WCS of the image to render.
        mjd : float
            MJD of the midpoint of the exposure.
        exptime : float
            Exposure time.
        xsize : int
            Size in pixels of CCD in x-direction.
        ysize : int
            Size in pixels of CCD in y-direction.
        obj_types : list-like [None]
            List or tuple of object types to render, e.g., ('star', 'galaxy').
            If None, then consider all object types.
        edge_pix : float [100]
            Size in pixels of the buffer region around nominal image
            to consider objects.
        logger : logging.Logger [None]
            Logger object.
        """
        self.file_name = file_name
        self.wcs = wcs
        self.mjd = mjd
        self.exptime = exptime
        self.bandpass = bandpass
        if xsize is not None:
            self.xsize = xsize
        else:
            self.xsize = roman.n_pix
        if ysize is not None:
            self.ysize = ysize
        else:
            self.ysize = roman.n_pix
        self.obj_types = obj_types
        self.edge_pix = edge_pix
        self.logger = galsim.config.LoggerWrapper(logger)

        if obj_types is not None:
            self.logger.warning(f"Object types restricted to {obj_types}")
        self.sca_center = wcs.toWorld(galsim.PositionD(self.xsize / 2.0, self.ysize / 2.0))
        self._objects = None

        # import os, psutil
        # process = psutil.Process()
        # print('skycat init',process.memory_info().rss)

    @property
    def objects(self):
        from packaging.version import Version
        from skycatalogs import __version__ as skycatalogs_version
        from skycatalogs import skyCatalogs

        if Version(skycatalogs_version) < Version("2.0"):
            PolygonalRegion = skyCatalogs.PolygonalRegion
        else:
            from skycatalogs.utils import PolygonalRegion

        if self._objects is None:
            # import os, psutil
            # process = psutil.Process()
            # print('skycat obj 1',process.memory_info().rss)
            # Select objects from polygonal region bounded by CCD edges
            corners = (
                (-self.edge_pix, -self.edge_pix),
                (self.xsize + self.edge_pix, -self.edge_pix),
                (self.xsize + self.edge_pix, self.ysize + self.edge_pix),
                (-self.edge_pix, self.ysize + self.edge_pix),
            )
            vertices = []
            for x, y in corners:
                sky_coord = self.wcs.toWorld(galsim.PositionD(x, y))
                vertices.append((sky_coord.ra / galsim.degrees, sky_coord.dec / galsim.degrees))
            region = PolygonalRegion(vertices)
            sky_cat = skyCatalogs.open_catalog(self.file_name)
            self._objects = sky_cat.get_objects_by_region(region, obj_type_set=self.obj_types, mjd=self.mjd)
            if not self._objects:
                self.logger.warning("No objects found on image.")
            # import os, psutil
            # process = psutil.Process()
            # print('skycat obj 2',process.memory_info().rss)
        return self._objects

    def get_sca_center(self):
        """
        Return the SCA center.
        """
        return self.sca_center

    def getNObjects(self):
        """
        Return the number of GSObjects to render
        """
        return len(self.objects)

    def getApproxNObjects(self):
        """
        Return the approximate number of GSObjects to render, as set in
        the class initializer.
        """
        return self.getNObjects()

    def getWorldPos(self, index):
        """
        Return the sky coordinates of the skyCatalog object
        corresponding to the specified index.

        Parameters
        ----------
        index : int
            Index of the (object_index, subcomponent) combination.

        Returns
        -------
        galsim.CelestialCoord
        """
        skycat_obj = self.objects[index]
        ra, dec = skycat_obj.ra, skycat_obj.dec
        return galsim.CelestialCoord(ra * galsim.degrees, dec * galsim.degrees)

    def getObj(self, index, gsparams=None, rng=None, exptime=30):
        """
        Return the galsim object for the skyCatalog object
        corresponding to the specified index.  If the skyCatalog
        object is a galaxy, the returned galsim object will be
        a galsim.Sum.

        Parameters
        ----------
        index : int
            Index of the object in the self.objects catalog.

        Returns
        -------
        galsim.GSObject
        """
        if not self.objects:
            raise RuntimeError("Trying to get an object from an empty sky catalog")

        faint = False
        skycat_obj = self.objects[index]
        gsobjs = skycat_obj.get_gsobject_components(gsparams)

        # Compute the flux or get the cached value.
        flux = (
            skycat_obj.get_roman_flux(self.bandpass.name, mjd=self.mjd) * self.exptime * roman.collecting_area
        )
        if np.isnan(flux):
            return None

        # if True and skycat_obj.object_type == 'galaxy':
        #     # Apply DC2 dilation to the individual galaxy components.
        #     for component, gsobj in gsobjs.items():
        #         comp = component if component != 'knots' else 'disk'
        #         a = skycat_obj.get_native_attribute(f'size_{comp}_true')
        #         b = skycat_obj.get_native_attribute(f'size_minor_{comp}_true')
        #         scale = np.sqrt(a/b)
        #         gsobjs[component] = gsobj.dilate(scale)

        # Set up simple SED if too faint
        if flux < 40:
            faint = True
        if not faint:
            seds = skycat_obj.get_observer_sed_components(mjd=self.mjd)

        gs_obj_list = []
        for component in gsobjs:
            if faint:
                gsobjs[component] = gsobjs[component].evaluateAtWavelength(self.bandpass)
                gs_obj_list.append(gsobjs[component] * self._trivial_sed)
            else:
                if component in seds:
                    gs_obj_list.append(gsobjs[component] * seds[component])

        if not gs_obj_list:
            return None

        if len(gs_obj_list) == 1:
            gs_object = gs_obj_list[0]
        else:
            gs_object = galsim.Add(gs_obj_list)

        # This should catch both "star" and "gaia_star" objects
        if "star" in skycat_obj.object_type:
            # Cap (star) flux at 30M photons to avoid gross artifacts when trying
            # to draw the Roman PSF in finite time and memory
            flux_cap = 3e7
            if flux > flux_cap:
                flux = flux_cap

        # Give the object the right flux
        gs_object = gs_object.withFlux(flux, self.bandpass)
        gs_object.flux = flux

        # Get the object type
        if (skycat_obj.object_type == "diffsky_galaxy") | (skycat_obj.object_type == "galaxy"):
            gs_object.object_type = "galaxy"
        if skycat_obj.object_type in {"star", "gaia_star"}:
            gs_object.object_type = "star"
        if skycat_obj.object_type == "snana":
            gs_object.object_type = "transient"

        return gs_object


class SkyCatalogLoader(InputLoader):
    """
    Class to load SkyCatalogInterface object.
    """

    def getKwargs(self, config, base, logger):
        req = {"file_name": str, "exptime": float}
        opt = {
            "edge_pix": float,
            "obj_types": list,
            "mjd": float,
            "xsize": int,
            "ysize": int,
        }
        kwargs, safe = galsim.config.GetAllParams(config, base, req=req, opt=opt)
        wcs = galsim.config.BuildWCS(base["image"], "wcs", base, logger=logger)
        kwargs["wcs"] = wcs
        kwargs["logger"] = logger

        if "bandpass" not in config:
            base["bandpass"] = galsim.config.BuildBandpass(base["image"], "bandpass", base, logger=logger)[0]

        kwargs["bandpass"] = base["bandpass"]
        # Sky catalog object lists are created per CCD, so they are
        # not safe to reuse.
        safe = False
        return kwargs, safe


def SkyCatObj(config, base, ignore, gsparams, logger):
    """
    Build an object according to info in the sky catalog.
    """
    skycat = galsim.config.GetInputObj("sky_catalog", config, base, "SkyCatObj")

    # Ensure that this sky catalog matches the CCD being simulated by
    # comparing center locations on the sky.
    world_center = base["world_center"]
    sca_center = skycat.get_sca_center()
    sep = sca_center.distanceTo(base["world_center"]) / galsim.arcsec
    # Centers must agree to within at least 1 arcsec:
    if sep > 1.0:
        message = (
            "skyCatalogs selection and SCA center do not agree: \n"
            "skycat.sca_center: "
            f"{sca_center.ra/galsim.degrees:.5f}, "
            f"{sca_center.dec/galsim.degrees:.5f}\n"
            "world_center: "
            f"{world_center.ra/galsim.degrees:.5f}, "
            f"{world_center.dec/galsim.degrees:.5f} \n"
            f"Separation: {sep:.2e} arcsec"
        )
        raise RuntimeError(message)

    # Setup the indexing sequence if it hasn't been specified.  The
    # normal thing with a catalog is to just use each object in order,
    # so we don't require the user to specify that by hand.  We can do
    # it for them.
    galsim.config.SetDefaultIndex(config, skycat.getNObjects())

    req = {"index": int}
    opt = {"num": int}
    kwargs, safe = galsim.config.GetAllParams(config, base, req=req, opt=opt)
    index = kwargs["index"]

    rng = galsim.config.GetRNG(config, base, logger, "SkyCatObj")

    obj = skycat.getObj(index, gsparams=gsparams, rng=rng)
    base["object_id"] = skycat.objects[index].id

    return obj, safe


def SkyCatWorldPos(config, base, value_type):
    """Return a value from the object part of the skyCatalog"""
    skycat = galsim.config.GetInputObj("sky_catalog", config, base, "SkyCatWorldPos")

    # Setup the indexing sequence if it hasn't been specified.  The
    # normal thing with a catalog is to just use each object in order,
    # so we don't require the user to specify that by hand.  We can do
    # it for them.
    galsim.config.SetDefaultIndex(config, skycat.getNObjects())

    req = {"index": int}
    opt = {"num": int}
    kwargs, safe = galsim.config.GetAllParams(config, base, req=req, opt=opt)
    index = kwargs["index"]

    pos = skycat.getWorldPos(index)
    return pos, safe


RegisterInputType("sky_catalog", SkyCatalogLoader(SkyCatalogInterface, has_nobj=True))
RegisterObjectType("SkyCatObj", SkyCatObj, input_type="sky_catalog")
RegisterValueType("SkyCatWorldPos", SkyCatWorldPos, [galsim.CelestialCoord], input_type="sky_catalog")

# This class was modified from https://github.com/LSSTDESC/imSim/. License info follows:

# Copyright (c) 2016-2019, LSST Dark Energy Science Collaboration (DESC)
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of imSim nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
