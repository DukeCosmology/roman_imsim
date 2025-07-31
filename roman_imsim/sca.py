import galsim
import galsim.config
import galsim.roman as roman
import numpy as np
from astropy.time import Time
from galsim.config import RegisterImageType
from galsim.config.image_scattered import ScatteredImageBuilder
from galsim.image import Image
import asdf
from astropy import wcs as fits_wcs
from astropy.io.fits import Header
from astropy.modeling.models import (
    Shift, Polynomial2D, Pix2Sky_TAN, RotateNative2Celestial, Mapping)
from gwcs import coordinate_frames as cf
import astropy.units as u
import astropy.coordinates
import gwcs

class RomanSCAImageBuilder(ScatteredImageBuilder):

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
        logger.debug(
            "image %d: Building RomanSCA: image, obj = %d,%d",
            image_num,
            image_num,
            obj_num,
        )

        self.nobjects = self.getNObj(config, base, image_num, logger=logger)
        logger.debug("image %d: nobj = %d", image_num, self.nobjects)

        # These are allowed for Scattered, but we don't use them here.
        extra_ignore = [
            "image_pos",
            "world_pos",
            "stamp_size",
            "stamp_xsize",
            "stamp_ysize",
            "nobjects",
        ]
        req = {"SCA": int, "filter": str, "mjd": float, "exptime": float}
        opt = {
            "draw_method": str,
            "stray_light": bool,
            "thermal_background": bool,
            "reciprocity_failure": bool,
            "dark_current": bool,
            "nonlinearity": bool,
            "ipc": bool,
            "read_noise": bool,
            "sky_subtract": bool,
            "ignore_noise": bool,
        }
        params = galsim.config.GetAllParams(config, base, req=req, opt=opt, ignore=ignore + extra_ignore)[0]

        self.sca = params["SCA"]
        base["SCA"] = self.sca
        self.filter = params["filter"]
        self.mjd = params["mjd"]
        self.exptime = params["exptime"]

        self.ignore_noise = params.get("ignore_noise", False)
        # self.exptime = params.get('exptime', roman.exptime)  # Default is roman standard exposure time.
        self.stray_light = params.get("stray_light", False)
        self.thermal_background = params.get("thermal_background", False)
        self.reciprocity_failure = params.get("reciprocity_failure", False)
        self.dark_current = params.get("dark_current", False)
        self.nonlinearity = params.get("nonlinearity", False)
        self.ipc = params.get("ipc", False)
        self.read_noise = params.get("read_noise", False)
        self.sky_subtract = params.get("sky_subtract", False)

        # If draw_method isn't in image field, it may be in stamp.  Check.
        self.draw_method = params.get("draw_method", base.get("stamp", {}).get("draw_method", "auto"))

        # pointing = CelestialCoord(ra=params['ra'], dec=params['dec'])
        # wcs = roman.getWCS(world_pos        = pointing,
        #                         PA          = params['pa']*galsim.degrees,
        #                         date        = params['date'],
        #                         SCAs        = self.sca,
        #                         PA_is_FPA   = True
        #                         )[self.sca]

        # # GalSim expects a wcs in the image field.
        # config['wcs'] = wcs

        # If user hasn't overridden the bandpass to use, get the standard one.
        if "bandpass" not in config:
            base["bandpass"] = galsim.config.BuildBandpass(base["image"], "bandpass", base, logger=logger)

        return roman.n_pix, roman.n_pix

    # def getBandpass(self, filter_name):
    #     if not hasattr(self, 'all_roman_bp'):
    #         self.all_roman_bp = roman.getBandpasses()
    #     return self.all_roman_bp[filter_name]

    
    
    def wcs_from_fits_header(self, header):
        """Convert a FITS WCS to a GWCS.

        This function reads SIP coefficients from a FITS WCS and implements
        the corresponding gWCS WCS.
        Copied from romanisim/wcs.py

        Parameters
        ----------
        header : astropy.io.fits.header.Header
            FITS header

        Returns
        -------
        wcs : gwcs.wcs.WCS
            gwcs WCS corresponding to header
        """

        # NOTE: this function ignores table distortions

        def coeffs_to_poly(mat, degree):
            pol = Polynomial2D(degree=degree)
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    if 0 < i + j <= degree:
                        setattr(pol, f'c{i}_{j}', mat[i, j])
            return pol


        w = fits_wcs.WCS(header)
        ny, nx = header['NAXIS2'] + 1, header['NAXIS1'] + 1
        x0, y0 = w.wcs.crpix

        cd = w.wcs.piximg_matrix

        cfx, cfy = np.dot(cd, [w.sip.a.ravel(), w.sip.b.ravel()])
        a = np.reshape(cfx, w.sip.a.shape)
        b = np.reshape(cfy, w.sip.b.shape)
        a[1, 0] = cd[0, 0]
        a[0, 1] = cd[0, 1]
        b[1, 0] = cd[1, 0]
        b[0, 1] = cd[1, 1]

        polx = coeffs_to_poly(a, w.sip.a_order)
        poly = coeffs_to_poly(b, w.sip.b_order)

        # construct GWCS:
        det2sky = (
            (Shift(-x0) & Shift(-y0)) | Mapping((0, 1, 0, 1)) | (polx & poly)
            | Pix2Sky_TAN() | RotateNative2Celestial(*w.wcs.crval, header['LONPOLE'])
        )

        detector_frame = cf.Frame2D(name="detector", axes_names=("x", "y"),
                                    unit=(u.pix, u.pix))
        sky_frame = cf.CelestialFrame(
            reference_frame=getattr(astropy.coordinates, w.wcs.radesys).__call__(),
            name=w.wcs.radesys,
            unit=(u.deg, u.deg)
        )
        pipeline = [(detector_frame, det2sky), (sky_frame, None)]
        gw = gwcs.WCS(pipeline)
        gw.bounding_box = ((-0.5, nx - 0.5), (-0.5, ny - 0.5))

        return gw



    def writeASDF(self, config, base, image, path, include_raw_header=False):
        """
        Method to write the file to disk

        Parameters:
        -----------
        path (str): Output path and filename
        include_raw_header (bool): If `True` include a copy of the raw FITS header
           as a dictionary in the ASDF file.
        """
    
        print("self attrs:", [a for a in dir(self) if not a.startswith("_")])
        print("config keys:", list(config.keys()))
        print("base keys:  ", list(base.keys()))
        
        for key, val in image.header.items():
            print(f"{key} = {val}")
        print("Image array shape:", image.array.shape)
        print(image.array)
        
        tree = {}
        
        # Fill out the data, err, dq blocks
        tree['data'] = image.array
        # modify_image (inside detector_physics) is not implemented as a class yet 
        # no err and dq exist currently so set them to None
        tree['err'] = None   
        tree['dq'] = None  
        

        # Fill out the wcs block - this is very hard since we need to change from wcs to gwcs object stored in asdf
        # The config wcs is the configuration for building the wcs above
        # The base is galsim.GSFitsWCS constructed from config wcs using wcs.py, and copied to image.wcs
        
        #turn the Galsim header to a fits header for the wcs_from_fits_header function
        galsim_wcs_header = image.wcs.header           
        d = dict(galsim_wcs_header)                     
        wcs_header = Header(d)
        #lack the two qualities for the wcs_from_fits_header function: number of pixel in each coordinate of the image
        ny, nx = image.array.shape
        wcs_header['NAXIS1'] = nx
        wcs_header['NAXIS2'] = ny
        tree['wcs'] = self.wcs_from_fits_header(wcs_header)

        # check for catalogs
        #tree['catalogs'] = self.catalogs   # what is this? can't find this property anywhere

        # Populate metadata
        # The properties in self/base/image are the same and I choose to use image/self when possible. The config properties are parameters for execution, not final results - so don't use them.
        tree['meta'] = {}
        if include_raw_header:
            tree['meta']['raw_header'] = image.header
        tree['meta']['telescope'] = 'ROMAN'
        tree['meta']['instrument'] = 'WFI'
        tree['meta']['optical_element'] = self.filter
        # use image/base ra/dec - they are ra/dec per sca
        # the ra/dec of config is the global ra/dec that the telescope point to (which has 18 sca for roman and each have a offset)
        tree['meta']['ra_pointing'] = image.wcs.center.ra.rad
        tree['meta']['dec_pointing'] = image.wcs.center.dec.rad
        tree['meta']['zptmag'] = image.header['ZPTMAG']    #2.5 * np.log10(self.exptime * roman.collecting_area)
        tree['meta']['pa_fpa'] = True 
        tree['meta']['obs_date'] = Time(self.mjd, format="mjd").datetime.isoformat() 
        tree['meta']['mjd_obs'] = self.mjd
        tree['meta']['exp_time'] = self.exptime
        tree['meta']['nreads'] = 1
        # Constant gain = 1 (Troxel private comm.)
        tree['meta']['gain'] = 1.0
        
        # Here are something inside detector_physics.py that seems not implemented inside yaml yet
        #tree['meta']['detector'] = f'SCA{self.fitsheader["SCA_NUM"]:02d}' #detector is not implemented in yaml yet
        #tree['meta']['sky_mean'] = self.fitsheader['SKY_MEAN']  #detector is not implemented in yaml yet

        
        af = asdf.AsdfFile({'roman': tree})
        af.write_to(path)
        
        return None


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
        
        
        full_xsize = base["image_xsize"]
        full_ysize = base["image_ysize"]
        wcs = base["wcs"]

        full_image = Image(full_xsize, full_ysize, dtype=float)
        full_image.setOrigin(base["image_origin"])
        full_image.wcs = wcs
        full_image.setZero()
        
        full_image.header = galsim.FitsHeader()
        full_image.header["EXPTIME"] = self.exptime
        full_image.header["MJD-OBS"] = self.mjd
        full_image.header["DATE-OBS"] = Time(self.mjd, format="mjd").datetime.isoformat()
        full_image.header["FILTER"] = self.filter
        full_image.header["ZPTMAG"] = 2.5 * np.log10(self.exptime * roman.collecting_area)

        base["current_image"] = full_image

        if "image_pos" in config and "world_pos" in config:
            raise galsim.GalSimConfigValueError(
                "Both image_pos and world_pos specified for Scattered image.",
                (config["image_pos"], config["world_pos"]),
            )

        if "image_pos" not in config and "world_pos" not in config:
            xmin = base["image_origin"].x
            xmax = xmin + full_xsize - 1
            ymin = base["image_origin"].y
            ymax = ymin + full_ysize - 1
            config["image_pos"] = {
                "type": "XY",
                "x": {"type": "Random", "min": xmin, "max": xmax},
                "y": {"type": "Random", "min": ymin, "max": ymax},
            }

        nbatch = self.nobjects // 1000 + 1
        for batch in range(nbatch):
            #start id of objects in this batch
            start_obj_num = self.nobjects * batch // nbatch
            #end id of objects in this batch
            end_obj_num = self.nobjects * (batch + 1) // nbatch
            #no of obj in batch
            nobj_batch = end_obj_num - start_obj_num
            if nbatch > 1:
                logger.warning(
                    "Start batch %d/%d with %d objects [%d, %d)",
                    batch + 1,
                    nbatch,
                    nobj_batch,
                    start_obj_num,
                    end_obj_num,
                )
            stamps, current_vars = galsim.config.BuildStamps(
                nobj_batch, base, logger=logger, obj_num=start_obj_num, do_noise=False
            )
            base["index_key"] = "image_num"

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

                logger.debug("image %d: full bounds = %s", image_num, str(full_image.bounds))
                logger.debug(
                    "image %d: stamp %d bounds = %s",
                    image_num,
                    k + start_obj_num,
                    str(stamps[k].bounds),
                )
                logger.debug("image %d: Overlap = %s", image_num, str(bounds))
                #imprint the stemp of each object in this loop
                full_image[bounds] += stamps[k][bounds]
            stamps = None

        # # Bring the image so far up to a flat noise variance
        # current_var = FlattenNoiseVariance(
        #         base, full_image, stamps, current_vars, logger)

        #manage return
        self.full_image = full_image
        if self.writeASDF:
            self.writeASDF(config, base, full_image, 'one_image.asdf', include_raw_header=False)
        
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

        base["current_noise_image"] = base["current_image"]
        wcs = base["wcs"]
        bp = base["bandpass"]
        rng = galsim.config.GetRNG(config, base)
        logger.info("image %d: Start RomanSCA detector effects", base.get("image_num", 0))

        # Things that will eventually be subtracted (if sky_subtract) will have their expectation
        # value added to sky_image.  So technically, this includes things that aren't just sky.
        # E.g. includes dark_current and thermal backgrounds.
        sky_image = image.copy()
        sky_level = roman.getSkyLevel(bp, world_pos=wcs.toWorld(image.true_center))
        logger.debug("Adding sky_level = %s", sky_level)
        if self.stray_light:
            logger.debug("Stray light fraction = %s", roman.stray_light_fraction)
            sky_level *= 1.0 + roman.stray_light_fraction
        wcs.makeSkyImage(sky_image, sky_level)

        # The other background is the expected thermal backgrounds in this band.
        # These are provided in e-/pix/s, so we have to multiply by the exposure time.
        if self.thermal_background:
            tb = roman.thermal_backgrounds[self.filter] * self.exptime
            logger.debug("Adding thermal background: %s", tb)
            sky_image += roman.thermal_backgrounds[self.filter] * self.exptime

        # The image up to here is an expectation value.
        # Realize it as an integer number of photons.
        poisson_noise = galsim.noise.PoissonNoise(rng)
        if self.draw_method == "phot":
            logger.debug("Adding poisson noise to sky photons")
            sky_image1 = sky_image.copy()
            sky_image1.addNoise(poisson_noise)
            image.quantize()  # In case any profiles used InterpolatedImage, in which case
            # the image won't necessarily be integers.
            image += sky_image1
        else:
            logger.debug("Adding poisson noise")
            image += sky_image
            image.addNoise(poisson_noise)

        # Apply the detector effects here.  Not all of these are "noise" per se, but they
        # happen interspersed with various noise effects, so apply them all in this step.

        # Note: according to Gregory Mosby & Bernard J. Rauscher, the following effects all
        # happen "simultaneously" in the photo diodes: dark current, persistence,
        # reciprocity failure (aka CRNL), burn in, and nonlinearity (aka CNL).
        # Right now, we just do them in some order, but this could potentially be improved.
        # The order we chose is historical, matching previous recommendations, but Mosby and
        # Rauscher don't seem to think those recommendations are well-motivated.

        # TODO: Add burn-in and persistence here.

        if self.reciprocity_failure:
            logger.debug("Applying reciprocity failure")
            roman.addReciprocityFailure(image)

        if self.dark_current:
            dc = roman.dark_current * self.exptime
            logger.debug("Adding dark current: %s", dc)
            sky_image += dc
            dark_noise = galsim.noise.DeviateNoise(galsim.random.PoissonDeviate(rng, dc))
            image.addNoise(dark_noise)

        if self.nonlinearity:
            logger.debug("Applying classical nonlinearity")
            roman.applyNonlinearity(image)

        # Mosby and Rauscher say there are two read noises.  One happens before IPC, the other
        # one after.
        # TODO: Add read_noise1
        if self.ipc:
            logger.debug("Applying IPC")
            roman.applyIPC(image)

        if self.read_noise:
            logger.debug("Adding read noise %s", roman.read_noise)
            image.addNoise(galsim.GaussianNoise(rng, sigma=roman.read_noise))

        logger.debug("Applying gain %s", roman.gain)
        image /= roman.gain

        # Make integer ADU now.
        image.quantize()

        if self.sky_subtract:
            logger.debug("Subtracting sky image")
            sky_image /= roman.gain
            sky_image.quantize()
            image -= sky_image

def fitsheader_to_dict(self):
        """
        Method to convert the FITS header to a plain dictionary
        """
        if self.full_image.header is not None:
            hdr_out = {}
            for key, value in self.full_image.header.items(): #not sure if .items() is a valid method here
                hdr_out[key] = value
            return hdr_out
        else:
            raise ValueError('self.fitsheader is empty, \
                             please load the header first')


# Register this as a valid type
RegisterImageType("roman_sca", RomanSCAImageBuilder())
