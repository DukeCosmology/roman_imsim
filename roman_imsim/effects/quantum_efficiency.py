import os
import numpy as np
import fitsio as fio
import galsim
from roman_imsim.effects import roman_effects
import galsim.roman as roman
from .utils import sca_number_to_file

class quantum_efficiency(roman_effects):
    def __init__(self, params, base, logger, rng, rng_iter=None):
        super().__init__(params, base, logger, rng, rng_iter)
        
        self.model = getattr(self, self.params['model'])
        if self.model is None:
            self.logger.warning("%s hasn't been implemented yet, the simple model will be applied for %s"%(str(self.params['model']), str(self.__class__.__name__)))
            self.model = self.simple_model
        
    def lab_model(self, image):
        if self.sca_filepath is None:
            self.logger.warning("No QE data file provided; a default value of QE = 1 will be used.")
            return image
        
        self.df = fio.FITS(os.path.join(self.sca_filepath, sca_number_to_file[self.sca]))
        self.logger.warning("Lab measured model will be applied for quantum efficiency.")
        image.array[:,:] *= self.df['RELQE1'][:,:] #4096x4096 array
        return image