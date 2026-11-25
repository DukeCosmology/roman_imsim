"""
Interface to obtain objects from skyCatalogs.
"""

import warnings
from pathlib import Path

import galsim
import numpy as np
import romanisim.models as models
import h5py
import pandas as pd

from galsim.config import (
    InputLoader,
    RegisterInputType,
    RegisterObjectType,
    RegisterValueType,
)

_RUNTIME_DIFFSKY_OBJECT_TYPES = frozenset({"diffsky26_galaxy"})
_GALAXY_OBJECT_TYPES = frozenset({"galaxy", "diffsky_galaxy", "diffsky26_galaxy"})


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
        flux_cap=None,
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
        max_flux : None [deprecated]
            If object flux exceeds max_flux, the return None for that object.
            if max_flux is None, then don't apply a maximum flux cut.
            This parameter is deprecated and will be removed in future.
        flux_cap : dict [None]
            Optional map of object_type -> maximum flux (photons). Objects
            whose type appears in this dict have their flux set to the
            corresponding value.
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
            self.xsize = models.parameters.n_pix
        if ysize is not None:
            self.ysize = ysize
        else:
            self.ysize = models.parameters.n_pix
        self.obj_types = obj_types
        self.edge_pix = edge_pix
        if max_flux is not None:
            warnings.warn(
                "max_flux is deprecated and has no effect; "
                "This parameter will be removed in a future release.",
                DeprecationWarning,
                stacklevel=2,
            )
        self.max_flux = max_flux
        self.flux_cap = flux_cap
        self.logger = galsim.config.LoggerWrapper(logger)

        if obj_types is not None:
            self.logger.warning(f"Object types restricted to {obj_types}")
        self.sca_center = wcs.toWorld(galsim.PositionD(self.xsize / 2.0, self.ysize / 2.0))
        self._objects = None
        self._prefetched_sed_batch = None

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

    def prefetch_seds(self, start, stop):
        """
        Prefetch runtime SEDs for a half-open range of image objects.
        """
        batch_key = (int(start), int(stop))
        if batch_key == self._prefetched_sed_batch:
            return
        if not 0 <= start < len(self.objects):
            raise IndexError(
                f"Invalid SkyCatalog SED prefetch range [{start}, {stop}) " f"for {len(self.objects)} objects"
            )
        elif stop >= len(self.objects):
            self.logger.warning(
                f"SkyCatalog SED prefetch range [{start}, {stop}) exceeds the"
                f"number of objects in the catalog: {len(self.objects)}."
            )

        objects = self.objects[start:stop]
        diffsky_objects = [obj for obj in objects if obj.object_type in _RUNTIME_DIFFSKY_OBJECT_TYPES]
        if diffsky_objects and hasattr(diffsky_objects[0], "prefetch_seds"):
            if hasattr(diffsky_objects[0], "clear_prefetched_seds"):
                diffsky_objects[0].clear_prefetched_seds()
            self.logger.info(
                "Prefetching Diffsky SEDs for %d objects [%d, %d)",
                len(diffsky_objects),
                start,
                stop,
            )
            diffsky_objects[0].prefetch_seds(diffsky_objects)

        self._prefetched_sed_batch = batch_key

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
            skycat_obj.get_roman_flux(self.bandpass.name, mjd=self.mjd)
            * self.exptime
            * models.parameters.collecting_area
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

        # Cap flux for configured object types to avoid gross artifacts when
        # trying to draw the Roman PSF in finite time and memory.
        if self.flux_cap and skycat_obj.object_type in self.flux_cap:
            cap = self.flux_cap[skycat_obj.object_type]
            if flux > cap:
                flux = cap

        # Give the object the right flux
        gs_object = gs_object.withFlux(flux, self.bandpass)
        gs_object.flux = flux

        # Get the object type
        if skycat_obj.object_type in _GALAXY_OBJECT_TYPES:
            gs_object.object_type = "galaxy"
        if skycat_obj.object_type in {"star", "gaia_star"}:
            gs_object.object_type = "star"
        if skycat_obj.object_type == "snana":
            gs_object.object_type = "transient"

        return gs_object

    
    def getGenGal(self, index, gsparams=None, rng=None, exptime=30):
        """
        Return a diffusion-generated GalSim object matched to the
        morphology of the corresponding DiffSky26 galaxy.

        The matching is based on galaxy size, reduced-shear
        ellipticity amplitude, and position angle, using morphology
        measurements obtained with Photutils.

        If no generated galaxy satisfies the morphology matching
        tolerances, a random generated galaxy is selected.

        Parameters
        ----------
        index : int
            Index of the object in the self.objects catalog.
        gsparams : galsim.GSParams, optional
            GalSim accuracy parameters.
        rng : galsim.BaseDeviate, optional
            Random number generator used for the random fallback.
        exptime : float
            Exposure time, retained for compatibility with getObj.

        Returns
        -------
        galsim.GSObject or None

        Notes
        -----
        This is a preliminary morphology-only implementation.
        No additional flux normalization or SED is applied.

        DiffSky component half-light radii and Photutils moment
        radii are not identical observables. The radius proxy
        factor is not a scientifically validated calibration.

        Pixel and sky position angle conventions must also be
        validated before scientific use.
        """

        if not self.objects:
            raise RuntimeError(
                "Trying to get an object from an empty sky catalog"
            )

        skycat_obj = self.objects[index]

        # Preserve the original implementation for non-galaxy
        # objects, including stars and supernovae.
        if skycat_obj.object_type not in _GALAXY_OBJECT_TYPES:
            return self.getObj(
                index, gsparams=gsparams, rng=rng, exptime=exptime
            )

        # The current morphology matching is specific to DiffSky26.
        if skycat_obj.object_type != "diffsky26_galaxy":
            raise NotImplementedError(
                "GenGal currently supports diffsky26_galaxy only."
            )

        # -----------------------------------------------------
        # 1. Generated galaxy catalogs and matching parameters
        # -----------------------------------------------------

        roman_catalog_dir = Path(
            "/hpc/group/cosmology/ds772/RomanGenGal/GenGal_ParamCat/cat_v1/"
        )

        roman_hdf5_dir = Path(
            "/hpc/group/cosmology/ds772/RomanGenGal/GenGal_hdf5"
        )

        n_gpu = 4
        dataset_name = "stamps"

        # DiffSky component and reference matching band.
        component = "disk"
        band = "H"

        # Morphology conversion parameters.
        pixel_scale_arcsec = 0.1
        radius_proxy_factor = 1.0
        pa_offset_deg = 0.0
        pa_parity = 1

        # Morphology matching tolerances.
        max_frac_radius = 0.5
        max_delta_g = 0.15
        max_delta_pa_deg = 30.0
        circular_g_threshold = 0.03

        require_selected_radial = True

        # -----------------------------------------------------
        # 2. Extract and convert the DiffSky26 morphology
        # -----------------------------------------------------

        prefix = "disk" if component == "disk" else "spheroid"

        target_radius = float(
            skycat_obj.get_native_attribute(
                f"{prefix}HalfLightRadiusArcsec"
            )
        )

        target_g1 = float(
            skycat_obj.get_native_attribute(
                f"{prefix}Ellipticity1"
            )
        )

        target_g2 = float(
            skycat_obj.get_native_attribute(
                f"{prefix}Ellipticity2"
            )
        )

        # DiffSky26 uses the GalSim reduced-shear convention.
        target_g = np.hypot(target_g1, target_g2)

        target_pa = np.degrees(
            0.5 * np.arctan2(target_g2, target_g1)
        ) % 180.0

        # Check whether the target morphology is valid.
        target_valid = (
            np.all(np.isfinite([
                target_radius,
                target_g1,
                target_g2,
                target_g,
                target_pa,
            ]))
            and target_radius > 0
            and target_g < 1
        )

        # -----------------------------------------------------
        # 3. Load and cache the Photutils morphology catalogs
        # -----------------------------------------------------

        if not hasattr(self, "_gen_gal_catalog"):

            catalogs = []

            columns = [
                "stamp_index",
                "object_id",
                "band",
                "detected",
                "selected_radial",
                "Rm_pix",
                "ellipticity_moments",
                "orientation_rad",
            ]

            for gpu in range(n_gpu):

                csv_path = (
                    roman_catalog_dir
                    / f"all_stamps_all_bands_gpu{gpu}.csv"
                )

                catalog = pd.read_csv(
                    csv_path,
                    usecols=columns,
                )

                # Preserve the association with the original
                # HDF5 file.
                catalog["gpu"] = gpu

                # Select the reference matching band.
                catalog = catalog.loc[
                    catalog["band"].astype(str).str.upper()
                    == band.upper()
                ]

                # Keep detected galaxies.
                catalog = catalog.loc[
                    catalog["detected"].eq(True)
                ]

                # Keep centrally selected galaxies.
                if require_selected_radial:
                    catalog = catalog.loc[
                        catalog["selected_radial"].eq(True)
                    ]

                catalogs.append(catalog)

            # Combine the four Photutils catalogs.
            catalog = pd.concat(
                catalogs,
                ignore_index=True,
            )

            # Remove missing morphology measurements.
            catalog = catalog.dropna(
                subset=[
                    "stamp_index",
                    "Rm_pix",
                    "ellipticity_moments",
                    "orientation_rad",
                ]
            )

            # Remove non-physical morphology measurements.
            catalog = catalog.loc[
                (catalog["Rm_pix"] > 0)
                & (catalog["ellipticity_moments"] >= 0)
                & (catalog["ellipticity_moments"] < 1)
            ].copy()

            # -------------------------------------------------
            # 4. Convert the Photutils morphology
            # -------------------------------------------------

            # Convert the moment radius from pixels to arcsec.
            catalog["radius_proxy_arcsec"] = (
                catalog["Rm_pix"]
                * pixel_scale_arcsec
                * radius_proxy_factor
            )

            # Photutils ellipticity is defined as e = 1 - q.
            # Convert to reduced-shear ellipticity:
            # g = (1 - q) / (1 + q) = e / (2 - e).
            e = catalog["ellipticity_moments"].to_numpy(
                dtype=float
            )

            catalog["g"] = e / (2.0 - e)

            # Convert the Photutils pixel orientation to the
            # assumed DiffSky coordinate convention.
            theta = (
                pa_parity
                * catalog["orientation_rad"].to_numpy(dtype=float)
                + np.deg2rad(pa_offset_deg)
            )

            catalog["pa_deg"] = np.degrees(theta) % 180.0

            catalog["g1"] = (
                catalog["g"] * np.cos(2.0 * theta)
            )

            catalog["g2"] = (
                catalog["g"] * np.sin(2.0 * theta)
            )

            # Remove non-finite morphology measurements.
            valid = (
                np.isfinite(catalog["radius_proxy_arcsec"])
                & np.isfinite(catalog["g"])
                & np.isfinite(catalog["g1"])
                & np.isfinite(catalog["g2"])
                & np.isfinite(catalog["pa_deg"])
            )

            catalog = catalog.loc[valid].reset_index(drop=True)

            if catalog.empty:
                raise RuntimeError(
                    "No valid generated galaxies available."
                )

            # Cache the converted catalog for subsequent calls.
            self._gen_gal_catalog = catalog

        roman_catalog = self._gen_gal_catalog

        # -----------------------------------------------------
        # 5. Morphology matching
        # -----------------------------------------------------

        best_index = None

        if target_valid:

            radius = roman_catalog[
                "radius_proxy_arcsec"
            ].to_numpy()

            g = roman_catalog["g"].to_numpy()

            pa = roman_catalog["pa_deg"].to_numpy()

            # Relative radius difference.
            dr = (
                np.abs(radius - target_radius)
                / target_radius
            )

            # Absolute reduced-shear amplitude difference.
            dg = np.abs(g - target_g)

            # Position angle difference modulo 180 degrees.
            dpa = np.abs(
                (pa - target_pa + 90.0) % 180.0 - 90.0
            )

            # Ignore PA when either galaxy is nearly circular.
            ignore_pa = (
                np.minimum(g, target_g)
                < circular_g_threshold
            )

            # Apply the morphology matching tolerances.
            valid = (
                (dr <= max_frac_radius)
                & (dg <= max_delta_g)
                & (
                    ignore_pa
                    | (dpa <= max_delta_pa_deg)
                )
            )

            # If valid matches exist, select the candidate
            # with the lowest morphology matching score.
            if np.any(valid):

                score = (
                    (dr / max_frac_radius) ** 2
                    + (dg / max_delta_g) ** 2
                    + np.where(
                        ignore_pa,
                        0.0,
                        (dpa / max_delta_pa_deg) ** 2,
                    )
                )

                # Exclude candidates outside the tolerances.
                score[~valid] = np.inf

                best_index = int(np.argmin(score))

        # -----------------------------------------------------
        # 6. Random fallback if no morphology match is found
        # -----------------------------------------------------

        if best_index is None:

            # Use the GalSim random number generator to ensure
            # reproducibility within the simulation.
            if rng is None:
                raise ValueError(
                    "A GalSim RNG is required for random fallback."
                )

            uniform_rng = galsim.UniformDeviate(rng)

            best_index = int(
                uniform_rng() * len(roman_catalog)
            )

        # Retrieve the selected generated galaxy.
        best = roman_catalog.iloc[best_index]

        stamp_index = int(best["stamp_index"])
        gpu = int(best["gpu"])

        # -----------------------------------------------------
        # 7. Identify the HDF5 file and rendering band
        # -----------------------------------------------------

        hdf5_path = (
            roman_hdf5_dir
            / f"pub_100k_gpu{gpu}.hdf5"
        )

        # Assumed channel ordering: Y, J, H.
        band_map = {
            "Y106": 0,
            "J129": 1,
            "H158": 2,
        }

        current_band = self.bandpass.name

        if current_band not in band_map:
            raise NotImplementedError(
                f"No generated galaxy stamps available "
                f"for Roman band {current_band}"
            )

        band_index = band_map[current_band]

        # -----------------------------------------------------
        # 8. Retrieve the generated galaxy stamp
        # -----------------------------------------------------

        with h5py.File(hdf5_path, "r") as h5:

            dataset = h5[dataset_name]

            if dataset.ndim != 4 or dataset.shape[1] != 3:
                raise ValueError(
                    "Expected an HDF5 dataset with shape "
                    "(N, 3, height, width)."
                )

            if not 0 <= stamp_index < dataset.shape[0]:
                raise IndexError(
                    f"Invalid stamp index {stamp_index} "
                    f"in {hdf5_path}"
                )

            stamp = np.asarray(
                dataset[stamp_index, band_index],
                dtype=np.float64,
            )

        # Reject invalid generated images.
        if (
            stamp.ndim != 2
            or not np.all(np.isfinite(stamp))
            or stamp.sum() <= 0
        ):
            return None

        # -----------------------------------------------------
        # 9. Construct the GalSim object
        # -----------------------------------------------------

        image = galsim.Image(
            stamp,
            scale=pixel_scale_arcsec,
        )

        # TODO: Implement physical flux normalization,
        # unit conversion, and SED assignment.
        gs_object = galsim.InterpolatedImage(
            image,
            x_interpolant="lanczos5",
            gsparams=gsparams,
        )

        gs_object.object_type = "galaxy"

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
        # Per-object-type flux limits live under stamp.flux_cap in the config.
        flux_cap = base.get("stamp", {}).get("flux_cap")
        if flux_cap is not None:
            kwargs["flux_cap"] = flux_cap
        # Sky catalog object lists are created per CCD, so they are
        # not safe to reuse.
        safe = False
        return kwargs, safe


def SkyCatObj(config, base, ignore, gsparams, logger):
    """
    Build an object according to info in the sky catalog.
    """
    skycat = galsim.config.GetInputObj("sky_catalog", config, base, "SkyCatObj")

    # RomanSCAImageBuilder exposes the exact range that BuildStamps is about
    # to render. Prefetching here happens before the first object's SED lookup
    # and only once for the range.
    sed_batch = base.get("_roman_imsim_sed_batch")
    if sed_batch is not None:
        start, stop = sed_batch
        skycat.prefetch_seds(start=start, stop=stop)

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
    kwargs, safe = galsim.config.GetAllParams(config, base, req=req, opt=opt)
    index = kwargs["index"]

    rng = galsim.config.GetRNG(config, base, logger, "SkyCatObj")

    obj = skycat.getObj(index, gsparams=gsparams, rng=rng)
    base["object_id"] = skycat.objects[index].id

    return obj, safe


def SkyCatGenObj(config, base, ignore, gsparams, logger):
    """
    Build an object using a diffusion-generated galaxy
    from the HDF5 files.
    """

    skycat = galsim.config.GetInputObj(
        "sky_catalog", config, base, "GenGal"
    )

    # Ensure that the sky catalog matches the SCA.
    world_center = base["world_center"]
    sca_center = skycat.get_sca_center()
    sep = sca_center.distanceTo(world_center) / galsim.arcsec

    if sep > 1.0:
        raise RuntimeError(
            f"Sky catalog and SCA centers do not agree. "
            f"Separation: {sep:.2e} arcsec"
        )

    # Set up the indexing sequence.
    galsim.config.SetDefaultIndex(config, skycat.getNObjects())

    req = {"index": int}
    opt = {"num": int}

    kwargs, safe = galsim.config.GetAllParams(
        config, base, req=req, opt=opt, ignore=ignore
    )

    index = kwargs["index"]

    rng = galsim.config.GetRNG(config, base, logger, "GenGal")

    # Retrieve the diffusion-generated galaxy.
    obj = skycat.getGenGal(index, gsparams=gsparams, rng=rng)

    base["object_id"] = skycat.objects[index].id

    return obj, False


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
RegisterObjectType("GenGal", SkyCatGenObj, input_type="sky_catalog")

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
