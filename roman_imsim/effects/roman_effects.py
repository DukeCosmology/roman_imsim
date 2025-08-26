import galsim as galsim
import numpy as np
from .utils import get_pointing
import roman_imsim.effects as Effects


class RomanEffects(object):
    """
    Class to simulate non-idealities and noise of roman detector images.
    """

    def __init__(self, params, base, logger, rng, rng_iter=None):
        self.params = params
        self.base = base
        self.visit = int(self.base["input"]["obseq_data"]["visit"])
        self.sca = base["image"]["SCA"]
        self.filter = base["image"]["filter"]
        self.sca_filepath = base["image"]["sca_filepath"]
        self.rng_iter = rng_iter if rng_iter else self.visit * self.sca

        self.rng = rng
        self.noise = galsim.PoissonNoise(self.rng)
        self.rng_np = np.random.default_rng(self.rng_iter)
        self.pointing = get_pointing(self.base, self.visit, self.sca)
        self.exptime = self.pointing.exptime
        self.logger = logger

        self.force_cvz = False
        if "force_cvz" in self.base["image"]["wcs"]:
            if self.base["image"]["wcs"]["force_cvz"]:
                self.force_cvz = True

        self.save_diff = False
        if "save_diff" in self.base["image"]:
            self.save_diff = bool(self.base["image"]["save_diff"])
            if "diff_dir" in self.base["output"]:
                self.diff_dir = self.base["output"]["diff_dir"]
            else:
                self.diff_dir = self.base["output"]["dir"]

    def is_model_valid(self):
        self.model = getattr(self, self.params["model"], None)
        if self.model is None:
            self.logger.warning(
                "%s hasn't been implemented yet, the simple model will be applied for %s"
                % (str(self.params["model"]), str(self.__class__.__name__))
            )
            self.model = self.simple_model

    def simple_model(self, image):
        self.logger.info("Applying the default model...")
        return image

    def cross_refer(self, effect_name):
        if effect_name not in self.base["image"]["add_effects"]:
            try:
                effect = getattr(Effects, effect_name)(
                    {"model": "simple_model"}, self.base, self.logger, self.rng
                )
            except Exception as e:
                self.logger.warning(e)
                # self.logger.warning("Effect %s is not implemented!"%(effect_name))
                return None
        else:
            try:
                params = self.base["image"]["add_effects"][effect_name]
                effect = getattr(Effects, effect_name)(params, self.base, self.logger, self.rng)
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
            self.pre.write("bg.fits", dir=self.diff_dir)
        return

    def diff(self, msg, im=None, verbose=True):
        if self.save_diff:
            diff = im - self.pre
            diff.write("%s_diff.fits" % msg, dir=self.diff_dir)
            self.pre = im.copy()
            im.write("%s_cumul.fits" % msg, dir=self.diff_dir)
        return


class setup_sky(object):
    def __init__(self, base, logger, rng, rng_iter=None):
        self.base = base
        self.logger = logger
        self.rng = rng

        self.visit = int(self.base["input"]["obseq_data"]["visit"])
        self.sca = base["image"]["SCA"]
        self.pointing = get_pointing(self.base, self.visit, self.sca)

    def get_sky_image(self):
        bounds = galsim.BoundsI(xmin=1, ymin=1, xmax=4088, ymax=4088)
        self.sky_img = galsim.Image(bounds=bounds, wcs=self.pointing.WCS)
        self.pointing.WCS.makeSkyImage(self.sky_img, 0.0)

        bound_pad = galsim.BoundsI(xmin=1, ymin=1, xmax=4096, ymax=4096)
        im_pad = galsim.Image(bound_pad)
        im_pad.array[4:-4, 4:-4] = self.sky_img.array[:, :]

        effects_list = list(self.base["image"]["add_effects"])
        if "Background" not in effects_list:
            return self.sky_img

        bkg_idx = effects_list.index("Background")
        for i in range(bkg_idx, len(effects_list)):
            effect_name = effects_list[i]
            args = (self.base["image"]["add_effects"][effect_name], self.base, self.logger, self.rng)
            effect = getattr(Effects, effect_name)(*args)
            im_pad = effect.apply(image=im_pad)

        im_pad.quantize()
        # output 4088x4088 img in uint16
        self.sky_img = im_pad.array[4:-4, 4:-4]
        self.sky_img = galsim.Image(self.sky_img, dtype=np.uint16)

        return self.sky_img

    def save_sky_img(self, outdir=".", sky_img_name="sky_img.fits"):
        self.sky_img.write(sky_img_name, dir=outdir)
