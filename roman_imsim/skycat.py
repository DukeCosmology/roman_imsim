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
from galsim.errors import galsim_warn


def no_lensing(self):
    return 0.0, 0.0, 1.0


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
        chromaticity=True,
        skycat_lensing=True,
        galsim_shear=False,
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
        chromatic : bool [True]
            Whether to use the chromatic GSObjects from skyCatalogs. If False,
            don't use the SED information and uses the flux inegrated over the
            bandpass instead.
        skycat_lensing : bool [False]
            If True, then do not apply lensing to the objects from SkyCatalogs.
        galsim_shear : bool [False]
            Whether a shear is specified in the config file. Only here for the
            purpose of raising a warning if both skycat_lensing and galsim_shear
            are applied.
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

        self.chromaticity = chromaticity

        self._skycat_lensing = skycat_lensing
        if self._skycat_lensing and galsim_shear:
            galsim_warn(
                "A shear is applied on top of the SkyCatalog shearing. It is "
                "recommended to set skycat_lensing = False when applying an "
                "external shear."
            )

    @property
    def objects(self):

        if not self._skycat_lensing:
            from skycatalogs.objects.diffsky_object import DiffskyObject

            DiffskyObject.get_wl_params = no_lensing

        from skycatalogs import skyCatalogs
        from skycatalogs import __version__ as skycatalogs_version
        from packaging.version import Version

        if Version(skycatalogs_version) < Version("2.0"):
            PolygonalRegion = skyCatalogs.PolygonalRegion
        else:
            from skycatalogs.utils import PolygonalRegion

        if self._objects is None:
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
        return self._objects

    def _build_dtype_dict(self):
        self._dtype_dict = {}
        obj_types = []
        for coll in self._objects.get_collections():
            objects_type = coll._object_type_unique
            if objects_type in obj_types:
                continue
            col_names = list(coll.native_columns)
            for col_name in col_names:
                try:
                    # Some columns cannot be read in snana
                    np_type = coll.get_native_attribute(col_name).dtype.type()
                except Exception as e:
                    self.logger.warning(f"The column {col_name} could not be read from skyCatalog.")
                    continue
                if np_type is None:
                    py_type = str
                else:
                    py_type = type(np_type.astype(object))
                self._dtype_dict[col_name] = py_type

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

    def getFlux(self, index=None, skycat_obj=None, filter=None, mjd=None, exptime=None):
        """
        Return the flux associated to an object.

        Parameters
        ----------
        index : int
            Index of the object in the self.objects catalog. Either index or
            skycat_obj must be provided. [Default: None]
        skycat_obj : skyCatalogs object
            The skyCatalogs object for which the flux is computed. Either index
            or skycat_obj must be provided. [Default: None]
        filter : str, optional
            Name of the filter for which the flux is computed. If None, use the
            filter provided during initialization. [Default: None]
        mjd : float, optional
            Date of the observation in MJD format. If None, use the
            mjd provided during initialization. [Default: None]
        exptime : int or float, optional
            Exposure time of the observation. If None, use the
            exptime provided during initialization. [Default: None]

        Returns
        -------
        flux
            Computer flux at the given date for the requested exposure time and
            filter.
        """

        if filter is None:
            filter = self.bandpass.name
        if mjd is None:
            mjd = self.mjd
        if exptime is None:
            exptime = self.exptime

        if index is not None and skycat_obj is None:
            skycat_obj = self.objects[index]
        elif skycat_obj is not None and index is None:
            pass
        else:
            raise ValueError("Either index or skycat_obj must be provided, but not both.")

        # We cache the SEDs for potential later use
        if hasattr(skycat_obj, "get_wl_params"):
            # _, _, mu = skycat_obj.get_wl_params()
            gamma1 = skycat_obj.get_native_attribute("shear1")
            gamma2 = skycat_obj.get_native_attribute("shear2")
            kappa = skycat_obj.get_native_attribute("convergence")
            mu = 1.0 / ((1.0 - kappa) ** 2 - (gamma1**2 + gamma2**2))
        else:
            mu = 1.0

        self._seds = skycat_obj.get_observer_sed_components(mjd=mjd)
        fluxes = {}
        for cmp_name, sed in self._seds.items():
            raw_flux = sed.calculateFlux(self.bandpass)
            fluxes[cmp_name] = raw_flux * mu * exptime * roman.collecting_area

        return fluxes

    def getValue(self, index, field):
        """
        Return a skyCatalog value for the an object.

        Parameters
        ----------
        index : int
            Index of the object in the self.objects catalog.
        field : str
            Name of the field for which you want the value.

        Returns
        -------
        int or float or str or None
            The value associated to the field or None if the field do not exist.
        """

        skycat_obj = self.objects[index]

        if field not in self._dtype_dict:
            # We cannot raise an error because one could have a field for snana
            # in the config and we don't want to crash because there are no SN
            # in this particular image. We then default to False which might not
            # be the right type for the required column but we have no way of knowing
            # the correct type if the column do not exist.
            self.logger.warning(f"The field {field} was not found in skyCatalog.")
            return None
        elif field not in skycat_obj.native_columns:
            if self._dtype_dict[field] is int:
                # There are no "special value" for integer so we default to
                # hopefully something completely off
                return -9999
            elif self._dtype_dict[field] is float:
                return np.nan
            elif self._dtype_dict[field] is str:
                return None
        else:
            return skycat_obj.get_native_attribute(field)

    def getObj(self, index, gsparams=None, rng=None):
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
        fluxes = self.getFlux(skycat_obj=skycat_obj)
        flux = sum(fluxes.values())
        if np.isnan(flux):
            return None

        # Set up simple SED if too faint
        if flux < 40:
            faint = True

        # This should catch both "star" and "gaia_star" objects
        if "star" in skycat_obj.object_type:
            # Cap (star) flux at 30M photons to avoid gross artifacts when trying
            # to draw the Roman PSF in finite time and memory
            flux_cap = 3e7
            if flux > flux_cap:
                flux = flux_cap
                fluxes["this_object"] = flux_cap

        if self.chromaticity:
            if faint:
                seds = {cmp_name: self._trivial_sed for cmp_name in gsobjs}
            else:
                seds = skycat_obj.get_observer_sed_components(mjd=self.mjd)
        else:
            seds = {cmp_name: 1.0 for cmp_name in gsobjs}

        gs_obj_list = []
        for component in gsobjs:
            # Give the object the right flux
            gsobj = gsobjs[component] * seds[component]
            if self.chromaticity:
                gsobj = gsobj.withFlux(fluxes[component], self.bandpass)
            else:
                gsobj = gsobj.withFlux(fluxes[component])
            gs_obj_list.append(gsobj)

        if not gs_obj_list:
            return None

        if len(gs_obj_list) == 1:
            gs_object = gs_obj_list[0]
        else:
            gs_object = galsim.Add(gs_obj_list)

        # gs_object = gs_object.withFlux(flux, self.bandpass)
        if not hasattr(gs_object, "flux"):
            gs_object.flux = flux

        if (skycat_obj.object_type == "diffsky_galaxy") | (skycat_obj.object_type == "galaxy"):
            object_type = "galaxy"
        if skycat_obj.object_type in {"star", "gaia_star"}:
            object_type = "star"
        if skycat_obj.object_type == "snana":
            object_type = "transient"

        return gs_object, object_type


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
            "skycat_lensing": bool,
            "chromaticity": bool,
        }
        kwargs, safe = galsim.config.GetAllParams(config, base, req=req, opt=opt)
        wcs = galsim.config.BuildWCS(base["image"], "wcs", base, logger=logger)
        kwargs["wcs"] = wcs
        kwargs["logger"] = logger

        if "bandpass" not in config:
            base["bandpass"] = galsim.config.BuildBandpass(base["image"], "bandpass", base, logger=logger)[0]

        kwargs["bandpass"] = base["bandpass"]
        kwargs["galsim_shear"] = "shear" in base["gal"]
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
            f"{sca_center.ra / galsim.degrees:.5f}, "
            f"{sca_center.dec / galsim.degrees:.5f}\n"
            "world_center: "
            f"{world_center.ra / galsim.degrees:.5f}, "
            f"{world_center.dec / galsim.degrees:.5f} \n"
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
    kwargs, safe = galsim.config.GetAllParams(config, base, req=req, opt=opt, ignore=ignore)
    index = kwargs["index"]

    rng = galsim.config.GetRNG(config, base, logger, "SkyCatObj")

    obj, object_type = skycat.getObj(index, gsparams=gsparams, rng=rng)
    base["object_id"] = skycat.objects[index].id
    base["object_type"] = object_type

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


def SkyCatValue(config, base, value_type):
    """Return a value from the object part of the skyCatalog"""

    skycat = galsim.config.GetInputObj("sky_catalog", config, base, "SkyCatValue")

    # Setup the indexing sequence if it hasn't been specified.  The
    # normal thing with a catalog is to just use each object in order,
    # so we don't require the user to specify that by hand.  We can do
    # it for them.
    galsim.config.SetDefaultIndex(config, skycat.getNObjects())

    req = {"field": str, "index": int}
    opt = {"obs_kind": str}
    params, safe = galsim.config.GetAllParams(config, base, req=req, opt=opt)
    field = params["field"]
    index = params["index"]
    obs_kind = params.get("obs_kind", None)

    if field == "flux":
        if obs_kind is None:
            val = skycat.getFlux(index)
        else:
            pointing = galsim.config.GetInputObj("obseq_data", config, base, "OpSeqDataLoader")
            filter = pointing.get("filter", obs_kind=obs_kind)
            exptime = pointing.get("exptime", obs_kind=obs_kind)
            mjd = pointing.get("mjd", obs_kind=obs_kind)
            val = skycat.getFlux(index, filter=filter, exptime=exptime, mjd=mjd)
    else:
        val = skycat.getValue(index, field)

    return val, safe


RegisterInputType("sky_catalog", SkyCatalogLoader(SkyCatalogInterface, has_nobj=True))
RegisterObjectType("SkyCatObj", SkyCatObj, input_type="sky_catalog")
RegisterValueType("SkyCatWorldPos", SkyCatWorldPos, [galsim.CelestialCoord], input_type="sky_catalog")

# Here we have to provide None as a type otherwise Galsim complains but I don't know why..
RegisterValueType("SkyCatValue", SkyCatValue, [float, int, str, None], input_type="sky_catalog")


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
