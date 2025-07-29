import os
import numpy as np
import fitsio as fio
from roman_imsim.effects import roman_effects
from .utils import sca_number_to_file


class vtpe(roman_effects):
    def __init__(self, params, base, logger, rng, rng_iter=None):
        super().__init__(params, base, logger, rng, rng_iter)

        self.model = getattr(self, self.params['model'])
        if self.model is None:
            self.logger.warning("%s hasn't been implemented yet, the simple model will be applied for %s"%(
                str(self.params['model']), str(self.__class__.__name__)))
            self.model = self.simple_model

    def simple_model(self, image):
        self.logger.warning("No vertical trailing pixel effect will be applied.")
        return image

    def lab_model(self, image):
        """
        Apply vertical trailing pixel effect.
        The vertical trailing pixel effect (VTPE) is a non-linear effect that is
        related to readout patterns.
        Q'[j,i] = Q[j,i] + f(  Q[j,i] - Q[j-1, i]  ),
        where f( dQ ) = dQ ( a + b * ln(1 + |dQ|/dQ0) )
        Input
        im           : image
        VTPE[0,512,512]  : coefficient a binned in 8x8
        VTPE[1,512,512]  : coefficient a
        VTPE[2,512,512]  : coefficient dQ0
        """

        if self.sca_filepath is None:
            self.logger.warning("No VTPE data file provided; no VTPE will be applied.")
            return image
        self.df = fio.FITS(os.path.join(self.sca_filepath, sca_number_to_file[self.sca]))

        self.logger.warning("Lab measured model will be applied for VTPE.")

        # expand 512x512 arrays to 4096x4096
        t = np.zeros((4096, 512))
        for row in range(t.shape[0]):
            t[row, row//8] = 1
        a_vtpe = t.dot(self.df['VTPE'][0, :, :][0]).dot(t.T)
        # NaN check
        if np.isnan(a_vtpe).any():
            self.logger.warning("vtpe skipped due to NaN in file")
            return image
        b_vtpe = t.dot(self.df['VTPE'][1, :, :][0]).dot(t.T)
        dQ0 = t.dot(self.df['VTPE'][2, :, :][0]).dot(t.T)

        dQ = image.array - np.roll(image.array, 1, axis=0)
        dQ[0, :] *= 0

        image.array[:, :] += dQ * (a_vtpe + b_vtpe * np.log(1. + np.abs(dQ)/dQ0))
        return image
