import os
import numpy as np
import fitsio as fio
import galsim
import galsim.roman as roman
from roman_imsim.effects import roman_effects
import roman_imsim.effects as effects
from galsim.config import ParseValue
from .utils import sca_number_to_file, get_pointing

class persistence(roman_effects):
    def __init__(self, params, base, logger, rng, rng_iter=None):
        super().__init__(params, base, logger, rng, rng_iter)
        
        self.model = getattr(self, self.params['model'])
        if self.model is None:
            self.logger.warning("%s hasn't been implemented yet, the simple model will be applied for %s"%(str(self.params['model']), str(self.__class__.__name__)))
            self.model = self.simple_model
            
        p_list = np.array([get_pointing(self.base, i, self.sca) for i in range(self.visit - 10, self.visit)])
        dt_list = np.array([(self.pointing.date - p.date).total_seconds() for p in p_list])
        self.p_pers = p_list[ np.where((dt_list > 0) & (dt_list < self.pointing.exptime*10))]
        
    def simple_model(self, image):
        for p in self.p_pers:
            dt = (self.pointing.date - p.date).total_seconds() - self.pointing.exptime/2 ##avg time since end of exposures
            # self.base['output']['file_name']['items'] = [p.filter, p.visit, p.sca]
            # imfilename = ParseValue(self.base['output'], 'file_name', self.base, str)[0]
            imfilename = self.base['output']['file_name']['format']%(p.filter, p.visit, p.sca)
            fn = os.path.join(self.base['output']['dir'], imfilename)

            # [TODO]
            if not os.path.exists(fn):
                continue

            ## apply all the effects that occured before persistence on the previouse exposures
            ## since max of the sky background is of order 100, it is thus negligible for persistence
            bound_pad = galsim.BoundsI( xmin=1, ymin=1,
                                        xmax=4096, ymax=4096)
            x = galsim.Image(bound_pad)
            x.array[4:-4, 4:-4] = galsim.Image(fio.FITS(fn)[0].read()).array[:,:]
            
            recip_failure_param = self.base['image']['add_effects']['recip_failure']
            recip_failure = effects.recip_failure(recip_failure_param, self.base, self.logger, self.rng)
            x = recip_failure.apply(image = x)

            x.array.clip(0) ##remove negative stimulus

            image.array[:,:] += galsim.roman.roman_detectors.fermi_linear(x.array, dt) * self.pointing.exptime
            
        return image
    
    def lab_model(self, image):
        if self.sca_filepath is None:
            self.logger.warning("No persistence data file provided; no persistence effect will be applied.")
            return image
        
        self.logger.warning("Lab measured model will be applied for persistence effect.")
        self.df = fio.FITS(os.path.join(self.sca_filepath, sca_number_to_file[self.sca]))
        
        #setup parameters for persistence
        Q01 = self.df['PERSIST'].read_header()['Q01']
        Q02 = self.df['PERSIST'].read_header()['Q02']
        Q03 = self.df['PERSIST'].read_header()['Q03']
        Q04 = self.df['PERSIST'].read_header()['Q04']
        Q05 = self.df['PERSIST'].read_header()['Q05']
        Q06 = self.df['PERSIST'].read_header()['Q06']
        alpha = self.df['PERSIST'].read_header()['ALPHA']

        #iterate over previous exposures
        for p in self.p_pers:
            dt = (self.pointing.date - p.date).total_seconds() - self.pointing.exptime/2 ##avg time since end of exposures
            fac_dt = (self.pointing.exptime/2.) / dt  ##linear time dependence (approximate until we get t1 and Delat t of the data)
            # self.base['output']['file_name']['items'] = [p.filter, p.visit, p.sca]
            # imfilename = ParseValue(self.base['output'], 'file_name', self.base, str)[0]
            imfilename = self.base['output']['file_name']['format']%(p.filter, p.visit, p.sca)
            fn = os.path.join(self.base['output']['dir'], imfilename)

            # [TODO]
            if not os.path.exists(fn):
                continue

            ## apply all the effects that occured before persistence on the previouse exposures
            ## since max of the sky background is of order 100, it is thus negligible for persistence
            ## same for brighter fatter effect
            bound_pad = galsim.BoundsI( xmin=1, ymin=1,
                                        xmax=4096, ymax=4096)
            x = galsim.Image(bound_pad)
            x.array[4:-4, 4:-4] = galsim.Image(fio.FITS(fn)[0].read()).array[:,:]
            # x = self.qe(x).array[:,:]
            
            qe_param = self.base['image']['add_effects']['quantum_efficiency']
            qe = effects.quantum_efficiency(qe_param, self.base, self.logger, self.rng)
            x = qe.apply(image = x).array[:, :]

            x = x.clip(0.1) ##remove negative and zero stimulus

            ## Do linear interpolation
            a = np.zeros(x.shape)
            a += ((x < Q01)) * x/Q01
            a += ((x >= Q01) & (x < Q02)) * (Q02-x)/(Q02-Q01)
            image.array[:,:] += a * self.df['PERSIST'][0,:,:][0] * fac_dt


            a = np.zeros(x.shape)
            a += ((x >= Q01) & (x < Q02)) * (x-Q01)/(Q02-Q01)
            a += ((x >= Q02) & (x < Q03)) * (Q03-x)/(Q03-Q02)
            image.array[:,:] += a * self.df['PERSIST'][1,:,:][0] * fac_dt

            a = np.zeros(x.shape)
            a += ((x >= Q02) & (x < Q03)) * (x-Q02)/(Q03-Q02)
            a += ((x >= Q03) & (x < Q04)) * (Q04-x)/(Q04-Q03)
            image.array[:,:] += a * self.df['PERSIST'][2,:,:][0] * fac_dt

            a = np.zeros(x.shape)
            a += ((x >= Q03) & (x < Q04)) * (x-Q03)/(Q04-Q03)
            a += ((x >= Q04) & (x < Q05)) * (Q05-x)/(Q05-Q04)
            image.array[:,:] += a * self.df['PERSIST'][3,:,:][0] * fac_dt

            a = np.zeros(x.shape)
            a += ((x >= Q04) & (x < Q05)) * (x-Q04)/(Q05-Q04)
            a += ((x >= Q05) & (x < Q06)) * (Q06-x)/(Q06-Q05)
            image.array[:,:] += a * self.df['PERSIST'][4,:,:][0] * fac_dt

            a = np.zeros(x.shape)
            a += ((x >= Q05) & (x < Q06)) * (x-Q05)/(Q06-Q05)
            a += ((x >= Q06)) * (x/Q06)**alpha       ##avoid fractional power of negative values
            image.array[:,:] += a * self.df['PERSIST'][5,:,:][0] * fac_dt

        return image