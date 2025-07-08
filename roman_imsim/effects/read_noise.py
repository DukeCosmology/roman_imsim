import os
import numpy as np
import fitsio as fio
import galsim
import galsim.roman as roman
from roman_imsim.effects import roman_effects
import roman_imsim.effects as effects
from galsim.config import ParseValue
from .utils import sca_number_to_file, get_pointing

class read_noise(roman_effects):
    def __init__(self, params, base, logger, rng, rng_iter=None):
        super().__init__(params, base, logger, rng, rng_iter)
        
        self.model = getattr(self, self.params['model'])
        if self.model is None:
            self.logger.warning("%s hasn't been implemented yet, the simple model will be applied for %s"%(str(self.params['model']), str(self.__class__.__name__)))
            self.model = self.simple_model

    def simple_model(self, image):
        self.logger.warning("Simple model will be applied for read noise.")
        self.im_read = image.copy()
        image.addNoise(self.read_noise)
        self.im_read = image - self.im_read
        # self.sky.addNoise(self.read_noise)
        return image
    
    def lab_model(self, image):
        if self.sca_filepath is None:
            self.logger.warning("No read noise data file provided; no read noise will be applied.")
            return image
        self.df = fio.FITS(os.path.join(self.sca_filepath, sca_number_to_file[self.sca]))
        
        self.logger.warning("Lab measured model will be applied for read noise.")
        rdn = self.df['READ'][2,:,:].flatten()  #flattened 4096x4096 array
        self.im_read = self.rng_np.normal(loc=0., scale=rdn).reshape(image.array.shape).astype(image.dtype)
        image.array[:,:] += self.im_read
        return image