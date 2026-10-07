from importlib.metadata import version, PackageNotFoundError
import gc
import galsim
import galsim.config
import romanisim.models as models
import numpy as np
from astropy.time import Time
from galsim.config import RegisterImageType
from galsim.config.image_scattered import ScatteredImageBuilder
from galsim.image import Image


class RomanSCAImageBuilderBase(ScatteredImageBuilder):
    """Shared setup, detector image creation, positions, and batching for Roman SCAs."""

    def setup(self, config, base, image_num, obj_num, ignore, logger):
        """Do the initialization and setup for building the image.

        This figures out the size that the image will be, but doesn't actually build it yet.

        Parameters:
            config:     The configuration dict for the image field.
            base:       The base configuration dict.
            image_num:  The current image number.
            obj_num:    The first object number in the image.
            ignore:     A list of parameters that are allowed to be in config that we can
                        ignore here. i.e. it won't be an error if these parameters are present.
            logger:     If given, a logger object to log progress.

        Returns:
            xsize, ysize
        """
        logger.debug(
            "image %d: Building RomanSCA: image, obj = %d,%d",
            image_num,
            image_num,
            obj_num,
        )

        self.nobjects = self.getNObj(config, base, image_num, logger=logger)
        logger.debug("image %d: nobj = %d", image_num, self.nobjects)

        # These are allowed for Scattered, but we don't use them here.
        extra_ignore = [
            "image_pos",
            "world_pos",
            "stamp_size",
            "stamp_xsize",
            "stamp_ysize",
            "nobjects",
        ]
        req = {"SCA": int, "filter": str, "mjd": float, "exptime": float}
        opt = {
            "draw_method": str,
            "use_fft_bright": bool,
        }
        params = galsim.config.GetAllParams(config, base, req=req, opt=opt, ignore=ignore + extra_ignore)[0]

        self.sca = params["SCA"]
        base["SCA"] = self.sca
        self.filter = params["filter"]
        self.mjd = params["mjd"]
        self.exptime = params["exptime"]

        # If draw_method isn't in image field, it may be in stamp.  Check.
        self.draw_method = params.get("draw_method", base.get("stamp", {}).get("draw_method", "phot"))

        # If user hasn't overridden the bandpass to use, get the standard one.
        if "bandpass" not in config:
            base["bandpass"] = galsim.config.BuildBandpass(base["image"], "bandpass", base, logger=logger)

        return models.parameters.n_pix, models.parameters.n_pix

    def _makeImage(self, base):
        """Create the detector image and its common Roman metadata."""
        full_xsize = base["image_xsize"]
        full_ysize = base["image_ysize"]
        wcs = base["wcs"]

        full_image = Image(full_xsize, full_ysize, dtype=float)
        full_image.setOrigin(base["image_origin"])
        full_image.wcs = wcs

        full_image.setZero()

        full_image.header = galsim.FitsHeader()
        try:
            full_image.header["VERSION"] = version("roman_imsim")
        except PackageNotFoundError:
            full_image.header["VERSION"] = "unknown"
        # Some of these should be in the WCS creation but it will be changed in the future so we leave it
        # here for now.
        full_image.header["NAXIS"] = 2
        full_image.header["NAXIS1"] = full_xsize
        full_image.header["NAXIS2"] = full_ysize
        full_image.header["RADESYS"] = "ICRS"
        full_image.header["SCA"] = self.sca
        # This should go to wcs header
        full_image.header["INSTRUME"] = "WFI"
        full_image.header["EXPTIME"] = self.exptime
        full_image.header["MJD-OBS"] = self.mjd
        full_image.header["DATE-OBS"] = Time(self.mjd, format="mjd").datetime.isoformat()
        full_image.header["FILTER"] = self.filter
        full_image.header["ZPTMAG"] = 2.5 * np.log10(self.exptime * models.parameters.collecting_area)

        base["current_image"] = full_image
        return full_image

    def _setupPositions(self, config, base):
        """Validate object positions or supply uniform detector positions."""
        full_xsize = base["image_xsize"]
        full_ysize = base["image_ysize"]
        if "image_pos" in config and "world_pos" in config:
            raise galsim.GalSimConfigValueError(
                "Both image_pos and world_pos specified for Scattered image.",
                (config["image_pos"], config["world_pos"]),
            )

        if "image_pos" not in config and "world_pos" not in config:
            xmin = base["image_origin"].x
            xmax = xmin + full_xsize - 1
            ymin = base["image_origin"].y
            ymax = ymin + full_ysize - 1
            config["image_pos"] = {
                "type": "XY",
                "x": {"type": "Random", "min": xmin, "max": xmax},
                "y": {"type": "Random", "min": ymin, "max": ymax},
            }

    def _iterObjectBatches(self):
        """Yield existing batch indices, counts, and half-open object ranges."""
        nbatch = self.nobjects // 1000 + 1
        for batch in range(nbatch):
            start_obj_num = self.nobjects * batch // nbatch
            end_obj_num = self.nobjects * (batch + 1) // nbatch
            yield batch, nbatch, start_obj_num, end_obj_num


class RomanSCAImageBuilder(RomanSCAImageBuilderBase):
    """Accumulate full-exposure image stamps using shared Roman SCA setup."""

    def buildImage(self, config, base, image_num, obj_num, logger):
        """Build an Image containing multiple objects placed at arbitrary locations.

        Parameters:
            config:     The configuration dict for the image field.
            base:       The base configuration dict.
            image_num:  The current image number.
            obj_num:    The first object number in the image.
            logger:     If given, a logger object to log progress.

        Returns:
            the final image and the current noise variance in the image as a tuple
        """
        full_image = self._makeImage(base)
        self._setupPositions(config, base)
        for batch, nbatch, start_obj_num, end_obj_num in self._iterObjectBatches():
            # Calculate the number of objects in this batch
            nobj_batch = end_obj_num - start_obj_num
            if nbatch > 1:
                logger.warning(
                    "Start batch %d/%d with %d objects [%d, %d)",
                    batch + 1,
                    nbatch,
                    nobj_batch,
                    start_obj_num,
                    end_obj_num,
                )
            stamps, current_vars = galsim.config.BuildStamps(
                nobj_batch, base, logger=logger, obj_num=start_obj_num, do_noise=False
            )
            base["index_key"] = "image_num"

            for k in range(nobj_batch):
                # This is our signal that the object was skipped.
                if stamps[k] is None:
                    continue
                bounds = stamps[k].bounds & full_image.bounds
                if not bounds.isDefined():  # pragma: no cover
                    # These noramlly show up as stamp==None, but technically it is possible
                    # to get a stamp that is off the main image, so check for that here to
                    # avoid an error.  But this isn't covered in the imsim test suite.
                    continue

                logger.debug("image %d: full bounds = %s", image_num, str(full_image.bounds))
                logger.debug(
                    "image %d: stamp %d bounds = %s",
                    image_num,
                    k + start_obj_num,
                    str(stamps[k].bounds),
                )
                logger.debug("image %d: Overlap = %s", image_num, str(bounds))
                # imprint the stemp of each object in this loop
                full_image[bounds] += stamps[k][bounds]
            stamps = None

        # # Bring the image so far up to a flat noise variance
        # current_var = FlattenNoiseVariance(
        #         base, full_image, stamps, current_vars, logger)

        return full_image, None


class RomanSCAImageBuilderCMOS(RomanSCAImageBuilderBase):
    """Accumulate interval photons and process reads using image.noise."""

    def buildImage(self, config, base, image_num, obj_num, logger):
        """Build an Image containing multiple objects placed at arbitrary locations.

        Parameters:
            config:     The configuration dict for the image field.
            base:       The base configuration dict.
            image_num:  The current image number.
            obj_num:    The first object number in the image.
            logger:     If given, a logger object to log progress.

        Returns:
            the final image and the current noise variance in the image as a tuple
        """
        rdata = galsim.config.GetInputObj("resultant_data", config, base, "ResultantDataLoader")
        strategy = rdata.get("strategy")
        full_image = self._makeImage(base)
        self._setupPositions(config, base)
        # Create index and lists for resultant management
        max_dt = strategy[-1][-1]
        resultant_i = 0
        resultant_buffer = []
        # Iterate through all dt
        if "_global" not in base:
            base["_global"] = {}
        if "stamp_setup_cache" not in base["_global"]:
            base["_global"]["stamp_setup_cache"] = {}

        for dt in np.arange(1, max_dt + 1):

            full_array = galsim.PhotonArray(0)

            for batch, nbatch, start_obj_num, end_obj_num in self._iterObjectBatches():
                nobj_batch = end_obj_num - start_obj_num
                if nbatch > 1:
                    logger.warning(
                        "Start batch %d/%d with %d objects [%d, %d)",
                        batch + 1,
                        nbatch,
                        nobj_batch,
                        start_obj_num,
                        end_obj_num,
                    )
                stamps, current_vars = galsim.config.BuildStamps(
                    nobj_batch, base, logger=logger, obj_num=start_obj_num, do_noise=False
                )
                # logger.warning(base["_global"]["stamp_setup_cache"][batch])
                base["index_key"] = "image_num"

                for k in range(nobj_batch):
                    # This is our signal that the object was skipped.
                    if stamps[k] is None:
                        continue
                    bounds = full_image.bounds  # stamps[k].bounds &
                    if not bounds.isDefined():  # pragma: no cover
                        # These noramlly show up as stamp==None, but technically it is possible
                        # to get a stamp that is off the main image, so check for that here to
                        # avoid an error.  But this isn't covered in the imsim test suite.
                        continue

                    # logger.debug("image %d: full bounds = %s", image_num, str(full_image.bounds))
                    # logger.debug(
                    #     "image %d: stamp %d bounds = %s",
                    #     image_num,
                    #     k + start_obj_num,
                    #     str(stamps[k].bounds),
                    # )
                    # logger.debug("image %d: Overlap = %s", image_num, str(bounds))
                    # full_image[bounds] += stamps[k][bounds]
                    # logger.warning(stamps[k])
                full_array = galsim.PhotonArray.concatenate([*stamps, full_array])

                stamps = None

            # # Bring the image so far up to a flat noise variance
            # current_var = FlattenNoiseVariance(
            #         base, full_image, stamps, current_vars, logger)
            # TODO : Apply BFE photon operation (uses current pre-read image and next photon array)

            # Turn full_image into running pre-read image
            full_array.addTo(full_image)
            del full_array
            gc.collect()
            # Decide what to do with readout based on resultant strategy
            if (
                np.array([item for sub in strategy for item in sub]) == dt
            ).any():  # does this dt exist in strategy
                if (np.array(strategy[resultant_i]) == dt).any():  # is it in our current resultant
                    readout_im = full_image.copy()
                    # Temporary full-exposure noise processing per read; interval
                    # backgrounds and read covariance are TODO.
                    galsim.config.AddNoise(base, readout_im, current_var=0, logger=logger)
                    resultant_buffer.extend([readout_im])
                    if len(resultant_buffer) > 1:
                        # combine readout images
                        resultant_buffer[0].array = resultant_buffer[0].array + resultant_buffer[1].array
                        del resultant_buffer[-1]
                        gc.collect()

                if np.array(strategy[resultant_i][-1]) == dt:  # is this dt the last in our current resultant
                    divisor = len(np.array(strategy[resultant_i]))
                    # divide summed images by the length of resultant to get the average
                    # apply headers to the image array
                    # TODO:apply header to the image array
                    resultant_buffer[0].array = resultant_buffer[0].array / divisor
                    resultant_buffer[0].write("resultant_{0}.fits".format(resultant_i))
                    resultant_i += 1
                    resultant_buffer = []
                    logger.warning("resultant{0} done".format(resultant_i))

        # full_array.write("photonarray.fits")
        # full_image.write("phot_image.fits")

        return full_image, None

    def addNoise(self, image, config, base, image_num, obj_num, current_var, logger):
        pass


# Register this as a valid type
RegisterImageType("roman_sca_cmos", RomanSCAImageBuilderCMOS())
# Register this as a valid type
RegisterImageType("roman_sca", RomanSCAImageBuilder())
