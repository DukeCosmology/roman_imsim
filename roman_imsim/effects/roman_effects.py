import galsim as galsim
import numpy as np
from .utils import get_pointing
import roman_imsim.effects as effects


class roman_effects(object):
    """
    Class to simulate non-idealities and noise of roman detector images.
    """

    def __init__(self, params, base, logger, rng, rng_iter=None):
        self.params = params
        self.base = base
        self.visit = int(self.base['input']['obseq_data']['visit'])
        self.sca = base['image']['SCA']
        self.filter = base['image']['filter']
        self.sca_filepath = base['image']['sca_filepath']
        self.rng_iter = rng_iter if rng_iter else self.visit * self.sca

        self.rng = rng
        self.noise = galsim.PoissonNoise(self.rng)
        self.rng_np = np.random.default_rng(self.rng_iter)
        self.pointing = get_pointing(self.base, self.visit, self.sca)
        self.exptime = self.pointing.exptime
        self.logger = logger

        self.force_cvz = False
        if 'force_cvz' in self.base['image']['wcs']:
            if self.base['image']['wcs']['force_cvz']:
                self.force_cvz = True

        self.save_diff = False
        if 'save_diff' in self.base['image']:
            self.save_diff = bool(self.base['image']['save_diff'])
            if 'diff_dir' in self.base['output']:
                self.diff_dir = self.base['output']['diff_dir']
            else:
                self.diff_dir = self.base['output']['dir']

    def simple_model(self, image):
        self.logger.info("Applying the default model...")
        return image

    def cross_refer(self, effect_name):
        if effect_name not in self.base['image']['add_effects']:
            try:
                effect = getattr(effects, effect_name)(
                    {'model': 'simple_model'}, self.base, self.logger, self.rng)
            except Exception as e:
                self.logger.warning(e)
                # self.logger.warning("Effect %s is not implemented!"%(effect_name))
                return None
        else:
            try:
                params = self.base['image']['add_effects'][effect_name]
                effect = getattr(effects, effect_name)(params, self.base, self.logger, self.rng)
            except Exception as e:
                self.logger.warning(e)
                # self.logger.warning("Effect %s is not implemented!"%(effect_name))
                return None
        return effect

    def apply(self, image):
        image = self.model(image)
        return image

    def set_diff(self, im=None):
        if self.save_diff:
            self.pre = im.copy()
            self.pre.write('bg.fits', dir=self.diff_dir)
        return

    def diff(self, msg, im=None, verbose=True):
        if self.save_diff:
            diff = im-self.pre
            diff.write('%s_diff.fits'%msg , dir=self.diff_dir)
            self.pre = im.copy()
            im.write('%s_cumul.fits'%msg, dir=self.diff_dir)
        return
