import os
import fitsio as fio
import galsim
import galsim.roman as roman
from . import RomanEffects
from .utils import sca_number_to_file


class DarkCurrent(RomanEffects):
    def __init__(self, params, base, logger, rng, rng_iter=None):
        super().__init__(params, base, logger, rng, rng_iter)

        self.model = getattr(self, self.params["model"], None)
        if self.model is None:
            self.logger.warning(
                "%s hasn't been implemented yet, the simple model will be applied for %s"
                % (str(self.params["model"]), str(self.__class__.__name__))
            )
            self.model = self.simple_model

    def simple_model(self, image):
        self.logger.warning("Simple model will be applied for dark current.")
        exptime = self.pointing.exptime
        self.dark_current_ = roman.dark_current * exptime
        self.im_dark = image.copy()
        dark_current_ = self.dark_current_
        dark_noise = galsim.DeviateNoise(galsim.PoissonDeviate(self.rng, dark_current_))
        image.addNoise(dark_noise)
        self.im_dark = image - self.im_dark
        return image

    def lab_model(self, image):
        if self.sca_filepath is None:
            self.logger.warning("No dark current file provided; no dark current will be applied.")
            return image
        self.df = fio.FITS(os.path.join(self.sca_filepath, sca_number_to_file[self.sca]))

        exptime = self.pointing.exptime
        self.logger.warning("Lab measured model will be applied for dark current.")
        self.dark_current_ = roman.dark_current * exptime + self.df["DARK"][:, :].flatten() * exptime
        dark_current_ = self.dark_current_.clip(0)
        # opt for numpy random geneator instead for speed
        self.im_dark = self.rng_np.poisson(dark_current_).reshape(image.array.shape).astype(image.dtype)
        image.array[:, :] += self.im_dark
        return image
