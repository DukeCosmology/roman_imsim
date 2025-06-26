import gc

import galsim
import galsim.config
import galsim.roman as roman
import numpy as np
from astropy.time import Time
from galsim.config import RegisterImageType
from galsim.config.image_scattered import ScatteredImageBuilder
from galsim.image import Image


class RomanSCAImageBuilder(ScatteredImageBuilder):

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
        # import os, psutil
        # process = psutil.Process()
        # print('sca setup 1',process.memory_info().rss)
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
        params = galsim.config.GetAllParams(config, base, req=req, opt=opt, ignore=ignore + extra_ignore)[0]

        self.sca = params["SCA"]
        base["SCA"] = self.sca
        self.filter = params["filter"]
        self.mjd = params["mjd"]
        self.exptime = params["exptime"]

        self.ignore_noise = params.get("ignore_noise", False)
        # self.exptime = params.get('exptime', roman.exptime)  # Default is roman standard exposure time.
        self.stray_light = params.get("stray_light", False)
        self.thermal_background = params.get("thermal_background", False)
        self.reciprocity_failure = params.get("reciprocity_failure", False)
        self.dark_current = params.get("dark_current", False)
        self.nonlinearity = params.get("nonlinearity", False)
        self.ipc = params.get("ipc", False)
        self.read_noise = params.get("read_noise", False)
        self.sky_subtract = params.get("sky_subtract", False)

        # If draw_method isn't in image field, it may be in stamp.  Check.
        self.draw_method = params.get("draw_method", base.get("stamp", {}).get("draw_method", "phot"))

        # pointing = CelestialCoord(ra=params['ra'], dec=params['dec'])
        # wcs = roman.getWCS(world_pos        = pointing,
        #                         PA          = params['pa']*galsim.degrees,
        #                         date        = params['date'],
        #                         SCAs        = self.sca,
        #                         PA_is_FPA   = True
        #                         )[self.sca]

        # # GalSim expects a wcs in the image field.
        # config['wcs'] = wcs

        # If user hasn't overridden the bandpass to use, get the standard one.
        if "bandpass" not in config:
            base["bandpass"] = galsim.config.BuildBandpass(base["image"], "bandpass", base, logger=logger)

        return roman.n_pix, roman.n_pix

    # def getBandpass(self, filter_name):
    #     if not hasattr(self, 'all_roman_bp'):
    #         self.all_roman_bp = roman.getBandpasses()
    #     return self.all_roman_bp[filter_name]

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
        full_xsize = base["image_xsize"]
        full_ysize = base["image_ysize"]
        wcs = base["wcs"]

        full_image = Image(full_xsize, full_ysize, dtype=float)
        full_image.setOrigin(base["image_origin"])
        full_image.wcs = wcs
        full_image.setZero()

        full_image.header = galsim.FitsHeader()
        full_image.header["EXPTIME"] = self.exptime
        full_image.header["MJD-OBS"] = self.mjd
        full_image.header["DATE-OBS"] = Time(self.mjd, format="mjd").datetime.isoformat()
        full_image.header["FILTER"] = self.filter
        full_image.header["ZPTMAG"] = 2.5 * np.log10(self.exptime * roman.collecting_area)

        base["current_image"] = full_image

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

        nbatch = self.nobjects // 1000 + 1
        for batch in range(nbatch):
            start_obj_num = self.nobjects * batch // nbatch
            end_obj_num = self.nobjects * (batch + 1) // nbatch
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
                full_image[bounds] += stamps[k][bounds]
            stamps = None

        # # Bring the image so far up to a flat noise variance
        # current_var = FlattenNoiseVariance(
        #         base, full_image, stamps, current_vars, logger)

        return full_image, None

    def addNoise(self, image, config, base, image_num, obj_num, current_var, logger):
        """Add the final noise to a Scattered image

        Parameters:
            image:          The image onto which to add the noise.
            config:         The configuration dict for the image field.
            base:           The base configuration dict.
            image_num:      The current image number.
            obj_num:        The first object number in the image.
            current_var:    The current noise variance in each postage stamps.
            logger:         If given, a logger object to log progress.
        """
        # check ignore noise
        if self.ignore_noise:
            return

        base["current_noise_image"] = base["current_image"]
        wcs = base["wcs"]
        bp = base["bandpass"]
        rng = galsim.config.GetRNG(config, base)
        logger.info("image %d: Start RomanSCA detector effects", base.get("image_num", 0))

        # Things that will eventually be subtracted (if sky_subtract) will have their expectation
        # value added to sky_image.  So technically, this includes things that aren't just sky.
        # E.g. includes dark_current and thermal backgrounds.
        sky_image = image.copy()
        sky_level = roman.getSkyLevel(bp, world_pos=wcs.toWorld(image.true_center))
        logger.debug("Adding sky_level = %s", sky_level)
        if self.stray_light:
            logger.debug("Stray light fraction = %s", roman.stray_light_fraction)
            sky_level *= 1.0 + roman.stray_light_fraction
        wcs.makeSkyImage(sky_image, sky_level)

        # The other background is the expected thermal backgrounds in this band.
        # These are provided in e-/pix/s, so we have to multiply by the exposure time.
        if self.thermal_background:
            tb = roman.thermal_backgrounds[self.filter] * self.exptime
            logger.debug("Adding thermal background: %s", tb)
            sky_image += roman.thermal_backgrounds[self.filter] * self.exptime

        # The image up to here is an expectation value.
        # Realize it as an integer number of photons.
        poisson_noise = galsim.noise.PoissonNoise(rng)
        if self.draw_method == "phot":
            logger.debug("Adding poisson noise to sky photons")
            sky_image1 = sky_image.copy()
            sky_image1.addNoise(poisson_noise)
            image.quantize()  # In case any profiles used InterpolatedImage, in which case
            # the image won't necessarily be integers.
            image += sky_image1
        else:
            logger.debug("Adding poisson noise")
            image += sky_image
            image.addNoise(poisson_noise)

        # Apply the detector effects here.  Not all of these are "noise" per se, but they
        # happen interspersed with various noise effects, so apply them all in this step.

        # Note: according to Gregory Mosby & Bernard J. Rauscher, the following effects all
        # happen "simultaneously" in the photo diodes: dark current, persistence,
        # reciprocity failure (aka CRNL), burn in, and nonlinearity (aka CNL).
        # Right now, we just do them in some order, but this could potentially be improved.
        # The order we chose is historical, matching previous recommendations, but Mosby and
        # Rauscher don't seem to think those recommendations are well-motivated.

        # TODO: Add burn-in and persistence here.

        if self.reciprocity_failure:
            logger.debug("Applying reciprocity failure")
            roman.addReciprocityFailure(image)

        if self.dark_current:
            dc = roman.dark_current * self.exptime
            logger.debug("Adding dark current: %s", dc)
            sky_image += dc
            dark_noise = galsim.noise.DeviateNoise(galsim.random.PoissonDeviate(rng, dc))
            image.addNoise(dark_noise)

        if self.nonlinearity:
            logger.debug("Applying classical nonlinearity")
            roman.applyNonlinearity(image)

        # Mosby and Rauscher say there are two read noises.  One happens before IPC, the other
        # one after.
        # TODO: Add read_noise1
        if self.ipc:
            logger.debug("Applying IPC")
            roman.applyIPC(image)

        if self.read_noise:
            logger.debug("Adding read noise %s", roman.read_noise)
            image.addNoise(galsim.GaussianNoise(rng, sigma=roman.read_noise))

        logger.debug("Applying gain %s", roman.gain)
        image /= roman.gain

        # Make integer ADU now.
        image.quantize()

        if self.sky_subtract:
            logger.debug("Subtracting sky image")
            sky_image /= roman.gain
            sky_image.quantize()
            image -= sky_image


class RomanSCAImageBuilderCMOS(ScatteredImageBuilder):

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
        # import os, psutil
        # process = psutil.Process()
        # print('sca setup 1',process.memory_info().rss)
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
        params = galsim.config.GetAllParams(config, base, req=req, opt=opt, ignore=ignore + extra_ignore)[0]

        self.sca = params["SCA"]
        base["SCA"] = self.sca
        self.filter = params["filter"]
        self.mjd = params["mjd"]
        self.exptime = params["exptime"]

        self.ignore_noise = params.get("ignore_noise", False)
        # self.exptime = params.get('exptime', roman.exptime)  # Default is roman standard exposure time.
        self.stray_light = params.get("stray_light", False)
        self.thermal_background = params.get("thermal_background", False)
        self.reciprocity_failure = params.get("reciprocity_failure", False)
        self.dark_current = params.get("dark_current", False)
        self.nonlinearity = params.get("nonlinearity", False)
        self.ipc = params.get("ipc", False)
        self.read_noise = params.get("read_noise", False)
        self.sky_subtract = params.get("sky_subtract", False)

        # If draw_method isn't in image field, it may be in stamp.  Check.
        self.draw_method = params.get("draw_method", base.get("stamp", {}).get("draw_method", "phot"))

        # pointing = CelestialCoord(ra=params['ra'], dec=params['dec'])
        # wcs = roman.getWCS(world_pos        = pointing,
        #                         PA          = params['pa']*galsim.degrees,
        #                         date        = params['date'],
        #                         SCAs        = self.sca,
        #                         PA_is_FPA   = True
        #                         )[self.sca]

        # # GalSim expects a wcs in the image field.
        # config['wcs'] = wcs

        # If user hasn't overridden the bandpass to use, get the standard one.
        if "bandpass" not in config:
            base["bandpass"] = galsim.config.BuildBandpass(base["image"], "bandpass", base, logger=logger)

        return roman.n_pix, roman.n_pix

    # def getBandpass(self, filter_name):
    #     if not hasattr(self, 'all_roman_bp'):
    #         self.all_roman_bp = roman.getBandpasses()
    #     return self.all_roman_bp[filter_name]

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
        full_xsize = base["image_xsize"]
        full_ysize = base["image_ysize"]
        wcs = base["wcs"]
        rdata = galsim.config.GetInputObj("resultant_data", config, base, "ResultantDataLoader")
        strategy = rdata.get("strategy")
        full_image = Image(full_xsize, full_ysize, dtype=float)
        full_image.setOrigin(base["image_origin"])
        full_image.wcs = wcs
        full_image.setZero()

        full_image.header = galsim.FitsHeader()
        full_image.header["EXPTIME"] = self.exptime
        full_image.header["MJD-OBS"] = self.mjd
        full_image.header["DATE-OBS"] = Time(self.mjd, format="mjd").datetime.isoformat()
        full_image.header["FILTER"] = self.filter
        full_image.header["ZPTMAG"] = 2.5 * np.log10(self.exptime * roman.collecting_area)

        base["current_image"] = full_image

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
        # Create index and lists for resultant management
        max_dt = strategy[-1][-1]
        resultant_i = 0
        resultant_buffer = []
        # Iterate through all dt
        for dt in np.arange(1, max_dt + 1):

            nbatch = self.nobjects // 1000 + 1
            full_array = galsim.PhotonArray(0)
            for batch in range(nbatch):
                start_obj_num = self.nobjects * batch // nbatch
                end_obj_num = self.nobjects * (batch + 1) // nbatch
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
                    readout_im = self.addNoiseToImage(readout_im, config, base, logger)
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

    def addNoiseToImage(self, image, config, base, logger):
        """Add the final noise to a Scattered image

        Parameters:
            image:          The image onto which to add the noise.
            config:         The configuration dict for the image field.
            base:           The base configuration dict.
            image_num:      The current image number.
            obj_num:        The first object number in the image.
            current_var:    The current noise variance in each postage stamps.
            logger:         If given, a logger object to log progress.
        """
        # check ignore noise
        if self.ignore_noise:
            return image

        base["current_noise_image"] = base["current_image"]
        wcs = base["wcs"]
        bp = base["bandpass"]
        rng = galsim.config.GetRNG(config, base)
        logger.info("image %d: Start RomanSCA detector effects", base.get("image_num", 0))

        # Things that will eventually be subtracted (if sky_subtract) will have their expectation
        # value added to sky_image.  So technically, this includes things that aren't just sky.
        # E.g. includes dark_current and thermal backgrounds.
        sky_image = image.copy()
        sky_level = roman.getSkyLevel(bp, world_pos=wcs.toWorld(image.true_center))
        logger.debug("Adding sky_level = %s", sky_level)
        if self.stray_light:
            logger.debug("Stray light fraction = %s", roman.stray_light_fraction)
            sky_level *= 1.0 + roman.stray_light_fraction
        wcs.makeSkyImage(sky_image, sky_level)

        # The other background is the expected thermal backgrounds in this band.
        # These are provided in e-/pix/s, so we have to multiply by the exposure time.
        if self.thermal_background:
            tb = roman.thermal_backgrounds[self.filter] * self.exptime
            logger.debug("Adding thermal background: %s", tb)
            sky_image += roman.thermal_backgrounds[self.filter] * self.exptime

        # The image up to here is an expectation value.
        # Realize it as an integer number of photons.
        poisson_noise = galsim.noise.PoissonNoise(rng)
        if self.draw_method == "phot":
            logger.debug("Adding poisson noise to sky photons")
            sky_image1 = sky_image.copy()
            sky_image1.addNoise(poisson_noise)
            image.quantize()  # In case any profiles used InterpolatedImage, in which case
            # the image won't necessarily be integers.
            image += sky_image1
        else:
            logger.debug("Adding poisson noise")
            image += sky_image
            image.addNoise(poisson_noise)

        # Apply the detector effects here.  Not all of these are "noise" per se, but they
        # happen interspersed with various noise effects, so apply them all in this step.

        # Note: according to Gregory Mosby & Bernard J. Rauscher, the following effects all
        # happen "simultaneously" in the photo diodes: dark current, persistence,
        # reciprocity failure (aka CRNL), burn in, and nonlinearity (aka CNL).
        # Right now, we just do them in some order, but this could potentially be improved.
        # The order we chose is historical, matching previous recommendations, but Mosby and
        # Rauscher don't seem to think those recommendations are well-motivated.

        # TODO: Add burn-in and persistence here.

        if self.reciprocity_failure:
            logger.debug("Applying reciprocity failure")
            roman.addReciprocityFailure(image)

        if self.dark_current:
            dc = roman.dark_current * self.exptime
            logger.debug("Adding dark current: %s", dc)
            sky_image += dc
            dark_noise = galsim.noise.DeviateNoise(galsim.random.PoissonDeviate(rng, dc))
            image.addNoise(dark_noise)

        if self.nonlinearity:
            logger.debug("Applying classical nonlinearity")
            roman.applyNonlinearity(image)

        # Mosby and Rauscher say there are two read noises.  One happens before IPC, the other
        # one after.
        # TODO: Add read_noise1
        if self.ipc:
            logger.debug("Applying IPC")
            roman.applyIPC(image)

        if self.read_noise:
            logger.debug("Adding read noise %s", roman.read_noise)
            image.addNoise(galsim.GaussianNoise(rng, sigma=roman.read_noise))

        logger.debug("Applying gain %s", roman.gain)
        image /= roman.gain

        # Make integer ADU now.
        image.quantize()

        if self.sky_subtract:
            logger.debug("Subtracting sky image")
            sky_image /= roman.gain
            sky_image.quantize()
            image -= sky_image
        return image

    def addNoise(self, image, config, base, image_num, obj_num, current_var, logger):
        pass


# Register this as a valid type
RegisterImageType("roman_sca_cmos", RomanSCAImageBuilderCMOS())
# Register this as a valid type
RegisterImageType("roman_sca", RomanSCAImageBuilder())
