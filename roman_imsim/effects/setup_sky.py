import numpy as np
import galsim
import roman_imsim.effects as effects
from .utils import get_pointing


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
        if "background" not in effects_list:
            return self.sky_img

        bkg_idx = effects_list.index("background")
        for i in range(bkg_idx, len(effects_list)):
            effect_name = effects_list[i]
            args = (self.base["image"]["add_effects"][effect_name], self.base, self.logger, self.rng)
            effect = getattr(effects, effect_name)(*args)
            im_pad = effect.apply(image=im_pad)

        im_pad.quantize()
        # output 4088x4088 img in uint16
        self.sky_img = im_pad.array[4:-4, 4:-4]
        self.sky_img = galsim.Image(self.sky_img, dtype=np.uint16)

        return self.sky_img

    def save_sky_img(self, outdir=".", sky_img_name="sky_img.fits"):
        self.sky_img.write(sky_img_name, dir=outdir)
