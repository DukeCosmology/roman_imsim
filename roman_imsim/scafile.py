import copy
import os
import warnings

import astropy.time
import galsim
import galsim.roman as roman
from galsim.config import OutputBuilder, RegisterOutputType


class SCABuilder(OutputBuilder):
    """Generate appropriate output in SCA file."""

    _added_eval_base_variables = False

    def setup(self, config, base, file_num, logger):
        """Do any necessary setup at the start of processing a file.

        Parameters:
            config:     The configuration dict for the output type.
            base:       The base configuration dict.
            file_num:   The current file_num.
            logger:     If given, a logger object to log progress.
        """
        # This is a copy of the base class code
        seed = galsim.config.SetupConfigRNG(base, logger=logger)
        logger.debug("file %d: seed = %d", file_num, seed)

        if "exptime" in config:
            base["exptime"] = galsim.config.ParseValue(config, "exptime", base, float)[0]
        else:
            base["exptime"] = roman.exptime

        # Save the detector size, so the input catalogs can use it to figure out which
        # objects will be visible.
        base["det_xsize"] = 4088
        base["det_ysize"] = 4088

    def getNFiles(self, config, base, logger=None):
        """Returns the number of files to be built.

        nfiles can be specified if you want.

        But the default is 189, not 1.

        Parameters:
            config:     The configuration dict for the output field.
            base:       The base configuration dict.

        Returns:
            the number of files to build.
        """
        return 1

    def buildImages(self, config, base, file_num, image_num, obj_num, ignore, logger):
        """Build the images for output.

        Parameters:
            config:     The configuration dict for the output field.
            base:       The base configuration dict.
            file_num:   The current file_num.
            image_num:  The current image_num.
            obj_num:    The current obj_num.
            ignore:     A list of parameters that are allowed to be in config that we can
                        ignore here.  i.e. it won't be an error if they are present.
            logger:     If given, a logger object to log progress.

        Returns:
            a list of the images built
        """
        # This is basically the same as the base class version.  Just a few extra things to
        # add to the ignore list.
        ignore += [
            "file_name",
            "dir",
            "nfiles",
            "det_num",
            "only_dets",
            "readout",
            "exptime",
            "camera",
        ]

        opt = {"cosmic_ray_rate": float, "cosmic_ray_catalog": str, "header": dict}
        params, safe = galsim.config.GetAllParams(config, base, opt=opt, ignore=ignore)

        image = galsim.config.BuildImage(base, image_num, obj_num, logger=logger)

        data_dir = ""
        # Add cosmic rays.
        cosmic_ray_rate = params.get("cosmic_ray_rate", 0)
        if cosmic_ray_rate > 0:
            cosmic_ray_catalog = params.get("cosmic_ray_catalog", None)
            if cosmic_ray_catalog is None:
                cosmic_ray_catalog = os.path.join(data_dir, "cosmic_rays_itl_2017.fits.gz")
            if not os.path.isfile(cosmic_ray_catalog):
                raise FileNotFoundError(f"{cosmic_ray_catalog} not found")

            logger.info(
                "Adding cosmic rays with rate %f using %s.",
                cosmic_ray_rate,
                cosmic_ray_catalog,
            )
            exptime = base["exptime"]

        # Add header keywords for various values written to the primary
        # header of the simulated raw output file, so that all the needed
        # information is in the eimage file.
        image.header = galsim.FitsHeader()
        exptime = base["exptime"]
        image.header["EXPTIME"] = exptime
        image.header["DET_NAME"] = self.det_name

        header_vals = copy.deepcopy(params.get("header", {}))

        # Helper function to parse a value with priority:
        # 1. from header_vals (popped from dict if present)
        # 2. from opsim_data
        # 3. specified default
        def parse(item, type, default):
            if item in header_vals:
                val = galsim.config.ParseValue(header_vals, item, base, type)[0]
                del header_vals[item]
            else:
                val = None
            return val

        # Get a few items needed more than once first
        mjd = parse("mjd", float, 51444.0)
        mjd_obs = parse("observationStartMJD", float, mjd)
        seqnum = parse("seqnum", int, 0)
        ratel = parse("fieldRA", float, 0.0)
        dectel = parse("fieldDec", float, 0.0)
        airmass = parse("airmass", float, "N/A")

        # Now construct the image header
        image.header["MJD"] = mjd
        image.header["MJD-OBS"] = mjd_obs, "Start of exposure"
        # NOTE: Should this day be the current day,
        # or the day at the time of the most recent noon?
        dayobs = astropy.time.Time(mjd_obs, format="mjd").strftime("%Y%m%d")
        image.header["DAYOBS"] = dayobs
        image.header["SEQNUM"] = seqnum
        image.header["CONTRLLR"] = "P", "simulated data"
        image.header["RUNNUM"] = parse("observationId", int, -999)
        image.header["OBSID"] = f"IM_P_{dayobs}_{seqnum:06d}"
        image.header["IMGTYPE"] = parse("image_type", str, "SKYEXP")
        image.header["REASON"] = parse("reason", str, "survey")
        image.header["RATEL"] = ratel
        image.header["DECTEL"] = dectel
        with warnings.catch_warnings():
            # Silence FITS warning about long header keyword
            warnings.simplefilter("ignore")
            image.header["ROTTELPOS"] = parse("rotTelPos", float, 0.0)
        image.header["FILTER"] = parse("band", str, "N/A/")
        image.header["CAMERA"] = base["output"]["camera"]
        image.header["AMSTART"] = airmass
        image.header["AMEND"] = airmass  # wrong, does anyone care?
        image.header["FOCUSZ"] = parse("focusZ", float, 0.0)

        # If there's anything left in header_vals, add it to the header.
        for k in header_vals:
            image.header[k] = galsim.config.ParseValue(header_vals, k, base, None)[0]

        return [image]


RegisterOutputType("LSST_CCD", SCABuilder())
