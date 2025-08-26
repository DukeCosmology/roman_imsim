import os
import fitsio as fio
from . import RomanEffects
from .utils import sca_number_to_file


class QuantumEfficiency(RomanEffects):
    def __init__(self, params, base, logger, rng, rng_iter=None):
        super().__init__(params, base, logger, rng, rng_iter)

        self.is_model_valid()

    def lab_model(self, image):
        if self.sca_filepath is None:
            self.logger.warning("No QE data file provided; a default value of QE = 1 will be used.")
            return image

        self.df = fio.FITS(os.path.join(self.sca_filepath, sca_number_to_file[self.sca]))
        self.logger.warning("Lab measured model will be applied for quantum efficiency.")
        image.array[:, :] *= self.df["RELQE1"][:, :]  # 4096x4096 array
        return image
