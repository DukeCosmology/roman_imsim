import os
import fitsio as fio
from . import RomanEffects
from .utils import sca_number_to_file


class Bias(RomanEffects):
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
        self.logger.warning("No bias will be applied.")
        return image

    def lab_model(self, image):
        if self.sca_filepath is None:
            self.logger.warning("No bias data provided; no bias will be applied.")
            return image
        self.df = fio.FITS(os.path.join(self.sca_filepath, sca_number_to_file[self.sca]))

        self.logger.warning("Lab measured model will be applied for bias.")
        bias = self.df["BIAS"][:, :]  # 4096x4096 img

        image.array[:, :] += bias
        return image
