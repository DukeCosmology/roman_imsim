import os
import numpy as np
import fitsio as fio
import galsim
from roman_imsim.effects import roman_effects
import galsim.roman as roman
from .utils import sca_number_to_file

class recip_failure(roman_effects):
    def __init__(self, params, base, logger, rng, rng_iter=None):
        super().__init__(params, base, logger, rng, rng_iter)
        
        self.alpha = self.params['alpha'] if 'alpha' in self.params else roman.reciprocity_alpha
        self.base_flux = self.params['base_flux'] if 'base_flux' in self.params else 1.0
        
        self.model = getattr(self, self.params['model'])
        if self.model is None:
            self.logger.warning("%s hasn't been implemented yet, the simple model will be applied for %s"%(str(self.params['model']), str(self.__class__.__name__)))
            self.model = self.simple_model
            
    def simple_model(self, image):
        # Add reciprocity effect
        exptime = self.pointing.exptime
        image.addReciprocityFailure(exp_time=exptime, alpha=self.alpha, base_flux=self.base_flux)
        return image