import os
import numpy as np
import fitsio as fio
from . import RomanEffects
from .utils import sca_number_to_file


class Saturation(RomanEffects):
    def __init__(self, params, base, logger, rng, rng_iter=None):
        super().__init__(params, base, logger, rng, rng_iter)

        self.model = getattr(self, self.params["model"], None)
        if self.model is None:
            self.logger.warning(
                "%s hasn't been implemented yet, the simple model will be applied for %s"
                % (str(self.params["model"]), str(self.__class__.__name__))
            )
            self.model = self.simple_model

        self.saturation_level = (
            self.params["saturation_level"] if "saturation_level" in self.params else 100000
        )

    def simple_model(self, image):
        self.logger.warning("Simple model will be applied for saturation.")
        saturation_array = np.ones_like(image.array) * self.saturation_level
        where_sat = np.where(image.array > saturation_array)
        image.array[where_sat] = saturation_array[where_sat]
        return image

    def lab_model(self, image):
        if self.sca_filepath is None:
            self.logger.warning("No saturation data file provided; no saturation effect will be applied.")
            return image

        self.logger.warning("Lab measured model will be applied for saturation effect.")
        self.df = fio.FITS(os.path.join(self.sca_filepath, sca_number_to_file[self.sca]))

        saturation_array = self.df["SATURATE"][:, :]  # 4096x4096 array
        where_sat = np.where(image.array > saturation_array)
        image.array[where_sat] = saturation_array[where_sat]
        return image
