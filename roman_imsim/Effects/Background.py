import os
import galsim
from . import RomanEffects
import galsim.roman as roman


class Background(RomanEffects):
    def __init__(self, params, base, logger, rng, rng_iter=None):
        super().__init__(params, base, logger, rng, rng_iter)
        self.thermal_background = (
            self.params["thermal_background"] if "thermal_background" in self.params else False
        )
        self.stray_light = self.params["stray_light"] if "stray_light" in self.params else False

        self.model = getattr(self, self.params["model"], None)
        if self.model is None:
            self.logger.warning(
                "%s hasn't been implemented yet, the simple model will be applied for %s"
                % (str(self.params["model"]), str(self.__class__.__name__))
            )
            self.model = self.simple_model

    def simple_model(self, image):
        if self.save_diff:
            orig = image.copy()

        self.logger.warning("Simple model will be applied for background.")
        pointing = self.pointing
        # Build current specification sky level if sky level not given
        if self.force_cvz:
            radec = self.translate_cvz(pointing.radec)
        else:
            radec = pointing.radec
        sky_level = roman.getSkyLevel(pointing.bpass, world_pos=radec, date=pointing.date)
        self.logger.debug("Adding sky_level = %s", sky_level)

        if self.stray_light:
            self.logger.debug("Stray light fraction = %s", roman.stray_light_fraction)
            sky_level *= 1.0 + roman.stray_light_fraction
        # Create sky image
        self.sky = galsim.Image(bounds=image.bounds, wcs=pointing.WCS)
        pointing.WCS.makeSkyImage(self.sky, sky_level)
        if self.thermal_background:
            tb = roman.thermal_backgrounds[pointing.filter] * pointing.exptime
            self.logger.debug("Adding thermal background: %s", tb)
            self.sky += tb
        self.sky.addNoise(self.noise)

        # [TODO] Not entirely sure about this block, since the 'auto' option is meant to
        # let the software choose which drawing method to use based on the total flux.
        if self.base["image"]["draw_method"] not in ["phot", "auto"]:
            image.addNoise(self.noise)

        # Adding sky level to the image.
        image += self.sky[self.sky.bounds & image.bounds]
        if self.save_diff:
            prev = image.copy()
            diff = prev - orig
            diff.write(os.path.join(self.diff_dir, "sky_a.fits"))
        return image
