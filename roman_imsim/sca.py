import galsim
import galsim.config
import galsim.roman as roman
import numpy as np
from astropy.time import Time
from galsim.config import RegisterImageType
from galsim.config.image_scattered import ScatteredImageBuilder
from galsim.image import Image


import roman_imsim.effects as RomanEffects


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
            "add_effects": dict,
            "sca_filepath": str,
            "sky_subtract": bool,
            "ignore_noise": bool,
            "save_diff": bool,
        }

        logger.warning("opt dict = %s" % (str(opt)))

        params = galsim.config.GetAllParams(config, base, req=req, opt=opt, ignore=ignore + extra_ignore)[0]

        self.sca = params["SCA"]
        base["SCA"] = self.sca
        self.filter = params["filter"]
        self.mjd = params["mjd"]
        self.exptime = params["exptime"]

        self.ignore_noise = params.get("ignore_noise", False)
        self.sky_subtract = params.get("sky_subtract", False)

        # If draw_method isn't in image field, it may be in stamp.  Check.
        self.draw_method = params.get("draw_method", base.get("stamp", {}).get("draw_method", "auto"))

        # pointing = CelestialCoord(ra=params['ra'], dec=params['dec'])
        # wcs = roman.getWCS(world_pos        = pointing,
        #                         PA          = params['pa']*galsim.degrees,
        #                         date        = params['date'],
        #                         SCAs        = self.sca,
        #                         PA_is_FPA   = True
        #                         )[self.sca]

        # # GalSim expects a wcs in the image field.
        # config['wcs'] = wcs

        self.rng = galsim.config.GetRNG(config, base)
        self.visit = int(base["input"]["obseq_data"]["visit"])

        self.sca_filepath = params.get("sca_filepath", None)

        # If user hasn't overridden the bandpass to use, get the standard one.
        if "bandpass" not in config:
            base["bandpass"] = galsim.config.BuildBandpass(base["image"], "bandpass", base, logger=logger)

        self.base = base
        self.logger = logger

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
        # rng = galsim.config.GetRNG(config, base)
        logger.info("image %d: Start RomanSCA detector effects", base.get("image_num", 0))

        # create padded image
        bound_pad = galsim.BoundsI(xmin=1, ymin=1, xmax=4096, ymax=4096)
        im_pad = galsim.Image(bound_pad)
        im_pad.array[4:-4, 4:-4] = image.array[:, :]

        effects_list = self.base["image"]["add_effects"].keys()
        for effect_name in effects_list:
            args = (self.base["image"]["add_effects"][effect_name], self.base, self.logger, self.rng)
            effect = getattr(RomanEffects, effect_name)(*args)
            im_pad = effect.apply(image=im_pad)

        im_pad.quantize()
        # output 4088x4088 img in uint16
        image.array[:, :] = im_pad.array[4:-4, 4:-4]

        if self.sky_subtract:
            logger.debug("Subtracting sky image")
            sky = RomanEffects.setup_sky(self.base, self.logger, self.rng)
            sky_image = sky.get_sky_image()
            image -= sky_image
            sky.save_sky_img(outdir=self.base["output"]["dir"])


# Register this as a valid type
RegisterImageType("roman_sca", RomanSCAImageBuilder())
