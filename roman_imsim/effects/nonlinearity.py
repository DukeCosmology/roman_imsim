import os
import fitsio as fio
import galsim.roman as roman
from . import RomanEffects
from .utils import sca_number_to_file


class Nonlinearity(RomanEffects):
    """
    Applying a quadratic non-linearity.

    Note that users who wish to apply some other nonlinearity function (perhaps for other NIR
    detectors, or for CCDs) can use the more general nonlinearity routine, which uses the
    following syntax:
    final_image.applyNonlinearity(NLfunc=NLfunc)
    with NLfunc being a callable function that specifies how the output image pixel values
    should relate to the input ones.
    """

    def __init__(self, params, base, logger, rng, rng_iter=None):
        super().__init__(params, base, logger, rng, rng_iter)

        self.is_model_valid()

    def simple_model(self, image):
        self.logger.info("Galsim.roman.NLfunc will be applied for simulating non-linearity effect.")
        image.applyNonlinearity(NLfunc=roman.NLfunc)
        return image

    def lab_model(self, image):
        if self.sca_filepath is None:
            self.logger.warning(
                "No non-linearity data file provided; no non-linearity effect will be applied."
            )
            return image

        self.logger.warning("Lab measured model will be applied for non-linearity effect.")

        self.df = fio.FITS(os.path.join(self.sca_filepath, sca_number_to_file[self.sca]))

        image.array[:, :] -= (
            self.df["CNL"][0, :, :][0] * image.array**2
            + self.df["CNL"][1, :, :][0] * image.array**3
            + self.df["CNL"][2, :, :][0] * image.array**4
        )

        return image
