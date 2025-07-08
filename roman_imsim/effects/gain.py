import os
import numpy as np
import fitsio as fio
import galsim
import galsim.roman as roman
from roman_imsim.effects import roman_effects
import roman_imsim.effects as effects
from galsim.config import ParseValue
from .utils import sca_number_to_file, get_pointing

class gain(roman_effects):
    def __init__(self, params, base, logger, rng, rng_iter=None):
        super().__init__(params, base, logger, rng, rng_iter)
        
        self.model = getattr(self, self.params['model'])
        if self.model is None:
            self.logger.warning("%s hasn't been implemented yet, the simple model will be applied for %s"%(str(self.params['model']), str(self.__class__.__name__)))
            self.model = self.simple_model

    def simple_model(self, image):
        self.logger.warning("Simple model will be applied for gain.")
        image /= roman.gain
        return image
    
    def lab_model(self, image):
        if self.sca_filepath is None:
            self.logger.warning("No gain data provided; galsim.roman.gain will be applied.")
            image /= roman.gain
            return image
        self.df = fio.FITS(os.path.join(self.sca_filepath, sca_number_to_file[self.sca]))
        
        self.logger.warning("Lab measured model will be applied for gain.")
        gain_map = self.df['GAIN'][:, :] #32x32 img

        t = np.zeros((4096, 32))
        for row in range(t.shape[0]):
            t[row, row//128] =1
        gain_expand = (t.dot(gain_map)).dot(t.T) #4096x4096 gain img
        image.array[:,:] /= gain_expand
        return image