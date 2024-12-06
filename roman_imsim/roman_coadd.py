import galsim
import galsim.roman as roman
import galsim.config
from galsim.config import RegisterImageType
from galsim.config import BuildStamps
from galsim.config.image import FlattenNoiseVariance
from galsim.config.image_scattered import ScatteredImageBuilder
from galsim.image import Image
from astropy.time import Time
from astropy.io import fits
import numpy as np

from .detector_effects import detector_effects


class RomanCoaddImageBuilder(ScatteredImageBuilder):

    def setup(self, config, base, image_num, obj_num, ignore, logger):
        """Do the initialization and setup for building the image.

        This figures out the size that the image will be, but doesn't actually build it yet.

        Parameters:
            config:     The configuration dict for the image field.
            base:       The base configuration dict.
            image_num:  The current image number.
            obj_num:    The first object number in the image.
            ignore:     A list of parameters that are allowed to be in config that we can
                        ignore here. i.e. it won't be an error if these parameters are present.
            logger:     If given, a logger object to log progress.

        Returns:
            xsize, ysize
        """
        # import os, psutil
        # process = psutil.Process()
        # print('sca setup 1',process.memory_info().rss)
        logger.debug('image %d: Building RomanSCA: image, obj = %d,%d',
                     image_num, image_num, obj_num)

        self.nobjects = self.getNObj(config, base, image_num, logger=logger)
        logger.debug('image %d: nobj = %d', image_num, self.nobjects)

        # These are allowed for Scattered, but we don't use them here.
        extra_ignore = ['image_pos', 'world_pos', 'stamp_size', 'stamp_xsize', 'stamp_ysize',
                        'nobjects']
        req = {
            'SCA': int,
            'filter': str,
            'mjd': float,
            'exptime': float,
            'coadd_file': str
        }
        opt = {
            'draw_method': str,
            # 'stray_light': bool,
            # 'thermal_background': bool,
            # 'reciprocity_failure': bool,
            # 'dark_current': bool,
            # 'nonlinearity': bool,
            # 'ipc': bool,
            # 'read_noise': bool,
            # 'sky_subtract': bool,
            'ignore_noise': bool,
            # 'sca_filepath': str,
            'dither_from_file': str,
            'save_diff': bool,
            'diff_dir': str,
            'xsize': int,
            'ysize': int
        }
        params = galsim.config.GetAllParams(
            config, base, req=req, opt=opt, ignore=ignore+extra_ignore)[0]

        self.sca = params['SCA']
        base['SCA'] = self.sca
        self.filter = params['filter']
        self.mjd = params['mjd']
        self.exptime = params['exptime']

        self.ignore_noise = params.get('ignore_noise', False)
        # self.exptime = params.get('exptime', roman.exptime)  # Default is roman standard exposure time.
        # self.stray_light = params.get('stray_light', False)
        # self.thermal_background = params.get('thermal_background', False)
        # self.reciprocity_failure = params.get('reciprocity_failure', False)
        # self.dark_current = params.get('dark_current', False)
        # self.nonlinearity = params.get('nonlinearity', False)
        # self.ipc = params.get('ipc', False)
        # self.read_noise = params.get('read_noise', False)
        # self.sky_subtract = params.get('sky_subtract', False)

        # If draw_method isn't in image field, it may be in stamp.  Check.
        self.draw_method = params.get('draw_method',
                                      base.get('stamp', {}).get('draw_method', 'auto'))

        # pointing = CelestialCoord(ra=params['ra'], dec=params['dec'])
        # wcs = roman.getWCS(world_pos        = pointing,
        #                         PA          = params['pa']*galsim.degrees,
        #                         date        = params['date'],
        #                         SCAs        = self.sca,
        #                         PA_is_FPA   = True
        #                         )[self.sca]

        # # GalSim expects a wcs in the image field.
        # config['wcs'] = wcs

        self.rng = galsim.config.GetRNG(config, base)
        self.visit = int(base['input']['obseq_data']['visit'])

        # self.sca_filepath = params.get('sca_filepath', None)
        # self.effects = detector_effects(params=base,
        #                                 visit=self.visit,
        #                                 sca=self.sca,
        #                                 filter=self.filter,
        #                                 logger=logger,
        #                                 rng=self.rng,
        #                                 rng_iter=self.visit * self.sca,
        #                                 sca_filepath=self.sca_filepath)

        # If user hasn't overridden the bandpass to use, get the standard one.
        if 'bandpass' not in config:
            base['bandpass'] = galsim.config.BuildBandpass(
                base['image'], 'bandpass', base, logger=logger)

        self.coadd_hdu = fits.open(params["coadd_file"])

        # return roman.n_pix, roman.n_pix
        return int(self.coadd_hdu[0].header['NAXIS1']), int(self.coadd_hdu[0].header['NAXIS2'])

    # def getBandpass(self, filter_name):
    #     if not hasattr(self, 'all_roman_bp'):
    #         self.all_roman_bp = roman.getBandpasses()
    #     return self.all_roman_bp[filter_name]

    def buildImage(self, config, base, image_num, obj_num, logger):
        """Build an Image containing multiple objects placed at arbitrary locations.

        Parameters:
            config:     The configuration dict for the image field.
            base:       The base configuration dict.
            image_num:  The current image number.
            obj_num:    The first object number in the image.
            logger:     If given, a logger object to log progress.

        Returns:
            the final image and the current noise variance in the image as a tuple
        """
        full_xsize = base['image_xsize']
        full_ysize = base['image_ysize']
        wcs = base['wcs']

        full_image = Image(full_xsize, full_ysize, dtype=float)
        full_image.setOrigin(base['image_origin'])
        full_image.wcs = wcs
        full_image.setZero()

        full_image.header = galsim.FitsHeader()
        full_image.header['EXPTIME'] = self.exptime
        full_image.header['MJD-OBS'] = self.mjd
        full_image.header['DATE-OBS'] = Time(self.mjd,
                                             format='mjd').datetime.isoformat()
        full_image.header['FILTER'] = self.filter
        full_image.header['ZPTMAG'] = 2.5 * \
            np.log10(self.exptime*roman.collecting_area)

        base['current_image'] = full_image

        if 'image_pos' in config and 'world_pos' in config:
            raise galsim.GalSimConfigValueError(
                "Both image_pos and world_pos specified for Scattered image.",
                (config['image_pos'], config['world_pos']))

        if 'image_pos' not in config and 'world_pos' not in config:
            xmin = base['image_origin'].x
            xmax = xmin + full_xsize-1
            ymin = base['image_origin'].y
            ymax = ymin + full_ysize-1
            config['image_pos'] = {
                'type': 'XY',
                'x': {'type': 'Random', 'min': xmin, 'max': xmax},
                'y': {'type': 'Random', 'min': ymin, 'max': ymax}
            }

        nbatch = self.nobjects // 1000 + 1
        for batch in range(nbatch):
            start_obj_num = (self.nobjects * batch // nbatch)
            end_obj_num = (self.nobjects * (batch+1) // nbatch)
            nobj_batch = end_obj_num - start_obj_num
            if nbatch > 1:
                logger.warning("Start batch %d/%d with %d objects [%d, %d)",
                               batch+1, nbatch, nobj_batch, start_obj_num, end_obj_num)
            stamps, current_vars = galsim.config.BuildStamps(
                nobj_batch, base, logger=logger, obj_num=start_obj_num, do_noise=False)
            base['index_key'] = 'image_num'

            for k in range(nobj_batch):
                # This is our signal that the object was skipped.
                if stamps[k] is None:
                    continue
                bounds = stamps[k].bounds & full_image.bounds
                if not bounds.isDefined():  # pragma: no cover
                    # These noramlly show up as stamp==None, but technically it is possible
                    # to get a stamp that is off the main image, so check for that here to
                    # avoid an error.  But this isn't covered in the imsim test suite.
                    continue

                logger.debug('image %d: full bounds = %s',
                             image_num, str(full_image.bounds))
                logger.debug('image %d: stamp %d bounds = %s',
                             image_num, k+start_obj_num, str(stamps[k].bounds))
                logger.debug('image %d: Overlap = %s', image_num, str(bounds))
                full_image[bounds] += stamps[k][bounds]
            stamps = None

            # # [TODO]
            # break

        # # Bring the image so far up to a flat noise variance
        # current_var = FlattenNoiseVariance(
        #         base, full_image, stamps, current_vars, logger)

        logger.info('roman pixel scale: %.5f' % (roman.pixel_scale))
        full_image /= (0.0390625/0.11)**2

        return full_image, None

    def addNoise(self, image, config, base, image_num, obj_num, current_var, logger):
        """Add the final noise to a Scattered image

        Parameters:
            image:          The image onto which to add the noise.
            config:         The configuration dict for the image field.
            base:           The base configuration dict.
            image_num:      The current image number.
            obj_num:        The first object number in the image.
            current_var:    The current noise variance in each postage stamps.
            logger:         If given, a logger object to log progress.
        """
        # check ignore noise
        if self.ignore_noise:
            return

        base['current_noise_image'] = base['current_image']
        wcs = base['wcs']
        bp = base['bandpass']
        # rng = galsim.config.GetRNG(config, base)
        logger.info('image %d: Start RomanSCA detector effects',
                    base.get('image_num', 0))

        # self.effects.setup_sky(image, force_cvz=self.effects.force_cvz,
        #                        stray_light=self.stray_light, thermal_background=self.thermal_background)
        # [TODO] quantize() at this step?

        # image = self.effects.add_background(
        #     image, draw_method=self.draw_method)

        # if self.sca_filepath is not None:
        #     # create padded image
        #     bound_pad = galsim.BoundsI(xmin=1, ymin=1,
        #                                xmax=4096, ymax=4096)
        #     im_pad = galsim.Image(bound_pad)
        #     im_pad.array[4:-4, 4:-4] = image.array[:, :]
        #     self.effects.set_diff(im_pad)
        #     im_pad = self.effects.qe(im_pad)
        #     self.effects.diff('qe', im_pad)

        #     im_pad = self.effects.bfe(im_pad)
        #     self.effects.diff('bfe', im_pad)

        #     im_pad = self.effects.add_persistence(im_pad)
        #     self.effects.diff('pers', im_pad)

        #     im_pad.quantize()
        #     self.effects.diff('quantize1', im_pad)

        #     im_pad = self.effects.dark_current(im_pad)
        #     self.effects.diff('dark', im_pad)

        #     im_pad = self.effects.saturate(im_pad)
        #     self.effects.diff('sat', im_pad)

        #     im_pad = self.effects.nonlinearity(im_pad)
        #     self.effects.diff('cnl', im_pad)

        #     im_pad = self.effects.interpix_cap(im_pad)
        #     self.effects.diff('ipc', im_pad)

        #     im_pad = self.effects.deadpix(im_pad)
        #     self.effects.diff('deadpix', im_pad)

        #     im_pad = self.effects.vtpe(im_pad)
        #     self.effects.diff('vtpe', im_pad)

        #     im_pad = self.effects.add_read_noise(im_pad)
        #     self.effects.diff('read', im_pad)

        #     im_pad = self.effects.add_gain(im_pad)
        #     self.effects.diff('gain', im_pad)

        #     im_pad = self.effects.add_bias(im_pad)
        #     self.effects.diff('bias', im_pad)

        #     im_pad.quantize()
        #     self.effects.diff('quantize2', im_pad)

        #     # output 4088x4088 img in uint16
        #     image.array[:, :] = im_pad.array[4:-4, 4:-4]

        #     # [TODO]
        #     # # data quality image
        #     # # 0x1 -> non-responsive
        #     # # 0x2 -> hot pixel
        #     # # 0x4 -> very hot pixel
        #     # # 0x8 -> adjacent to pixel with strange response
        #     # # 0x10 -> low CDS, high total noise pixel (may have strange settling behaviors, not recommended for precision applications)
        #     # # 0x20 -> CNL fit went down to the minimum number of points (remaining degrees of freedom = 0)
        #     # # 0x40 -> no solid-waffle solution for this region (set gain value to array median). normally occurs in a few small regions of some SCAs with lots of bad pixels. [recommend not to use these regions for WL analysis]
        #     # # 0x80 -> wt==0
        #     # dq = self.df['BADPIX'][4:4092, 4:4092]
        #     # # get weight map
        #     # if wt is not None:
        #     # dq[wt==0] += 128

        #     # sky_noise = self.sky.copy()
        #     # sky_noise = self.finalize_sky_im(sky_noise, pointing)

        # else:
        #     # Introduce reciprocity failure to image
        #     image = self.effects.recip_failure(image)
        #     image.quantize()  # At this point in the image generation process, an integer number of photons gets detected
        #     image = self.effects.dark_current(
        #         image)  # Add dark current to image
        #     image = self.effects.add_persistence(image)
        #     image = self.effects.saturate(image)
        #     image = self.effects.nonlinearity(image)  # Apply nonlinearity
        #     # Introduce interpixel capacitance to image.
        #     image = self.effects.interpix_cap(image)
        #     image = self.effects.add_read_noise(image)
        #     image = self.effects.e_to_ADU(image)  # Convert electrons to ADU

        # # Make integer ADU now.
        # image.quantize()

        # if self.sky_subtract:
        #     logger.debug("Subtracting sky image")
        #     sky_image = self.effects.finalize_sky_im(self.effects.sky.copy())
        #     image -= sky_image


# Register this as a valid type
RegisterImageType('roman_coadd', RomanCoaddImageBuilder())
