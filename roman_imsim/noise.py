import galsim
import galsim.roman as roman
from galsim.config import NoiseBuilder, RegisterNoiseType


class RomanNoiseBuilder(NoiseBuilder):

    def addNoise(self, config, base, image, rng, current_var, draw_method, logger):
        """Read the noise parameters from the config dict and add the appropriate noise to the
        given image.

        Parameters
        ----------
            config: dict
                The configuration dict for the noise field.
            base: dict
                The base configuration dict.
            im: galsim.Image
                The image onto which to add the noise
            rng: galsim.BaseDeviate
                The random number generator to use for adding the noise.
            current_var: float
                The current noise variance present in the image already.
            draw_method: str
                The method that was used to draw the objects on the image.
            logger: logging.Logger
                If given, a logger object to log progress.

        Returns
        -------
        var (None)
            the variance of the noise model (units are ADU if gain != 1)
            NOT IMPLEMENTED
        """

        opt = {
            "stray_light": bool,
            "thermal_background": bool,
            "reciprocity_failure": bool,
            "dark_current": bool,
            "nonlinearity": bool,
            "ipc": bool,
            "read_noise": bool,
            "sky_subtract": bool,
        }

        params, safe = galsim.config.GetAllParams(config, base, req={}, opt=opt, ignore=[])

        stray_light = params.get("stray_light", False)
        thermal_background = params.get("thermal_background", False)
        reciprocity_failure = params.get("reciprocity_failure", False)
        dark_current = params.get("dark_current", False)
        nonlinearity = params.get("nonlinearity", False)
        ipc = params.get("ipc", False)
        read_noise = params.get("read_noise", False)
        sky_subtract = params.get("sky_subtract", True)

        base["current_noise_image"] = base["current_image"]
        wcs = base["wcs"]
        bp = base["bandpass"]
        filter_name = bp.name
        exptime, _ = galsim.config.ParseValue(base["image"], "exptime", base, float)
        logger.info("image %d: Start RomanSCA detector effects", base.get("image_num", 0))

        # Things that will eventually be subtracted (if sky_subtract) will have their expectation
        # value added to sky_image.  So technically, this includes things that aren't just sky.
        # E.g. includes dark_current and thermal backgrounds.
        sky_image = image.copy()
        sky_level = roman.getSkyLevel(bp, world_pos=wcs.toWorld(image.true_center))
        logger.debug("Adding sky_level = %s", sky_level)
        if stray_light:
            logger.debug("Stray light fraction = %s", roman.stray_light_fraction)
            sky_level *= 1.0 + roman.stray_light_fraction
        wcs.makeSkyImage(sky_image, sky_level)

        # The other background is the expected thermal backgrounds in this band.
        # These are provided in e-/pix/s, so we have to multiply by the exposure time.
        if thermal_background:
            tb = roman.thermal_backgrounds[filter_name] * exptime
            logger.debug("Adding thermal background: %s", tb)
            sky_image += roman.thermal_backgrounds[filter_name] * exptime

        # The image up to here is an expectation value.
        # Realize it as an integer number of photons.
        poisson_noise = galsim.noise.PoissonNoise(rng)
        if draw_method == "phot":
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

        if reciprocity_failure:
            logger.debug("Applying reciprocity failure")
            roman.addReciprocityFailure(image)

        if dark_current:
            dc = roman.dark_current * exptime
            logger.debug("Adding dark current: %s", dc)
            sky_image += dc
            dark_noise = galsim.noise.DeviateNoise(galsim.random.PoissonDeviate(rng, dc))
            image.addNoise(dark_noise)

        if nonlinearity:
            logger.debug("Applying classical nonlinearity")
            roman.applyNonlinearity(image)

        # Mosby and Rauscher say there are two read noises. One happens before IPC, the other
        # one after.
        # TODO: Add read_noise1
        if ipc:
            logger.debug("Applying IPC")
            roman.applyIPC(image)

        if read_noise:
            logger.debug("Adding read noise %s", roman.read_noise)
            image.addNoise(galsim.GaussianNoise(rng, sigma=roman.read_noise))

        logger.debug("Applying gain %s", roman.gain)
        image /= roman.gain

        # Make integer ADU now.
        image.quantize()

        if sky_subtract:
            logger.debug("Subtracting sky image")
            sky_image /= roman.gain
            sky_image.quantize()
            image -= sky_image

        return None


RegisterNoiseType("RomanNoise", RomanNoiseBuilder())
