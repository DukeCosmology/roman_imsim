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
import sys
import os
#from .asdf_helper import mk_level2_image_with_wcs
from pathlib import Path
from roman_datamodels.datamodels import ImageModel

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



    def writeASDF_old(self, config, base, image, path, include_raw_header=False):
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
        
        
        print("\nconfig keys and values:")
        for k, v in config.items():
            if isinstance(v, np.ndarray):
                print(f"  {k} = ndarray, shape={v.shape}, dtype={v.dtype}")
            else:
                print(f"  {k} = {v}")

      
        print("\nbase keys and values:")
        for k, v in base.items():
            if isinstance(v, np.ndarray):
                print(f"  {k} = ndarray, shape={v.shape}, dtype={v.dtype}")
            else:
                print(f"  {k} = {v}")


        print("\nimage header:")
        for key, val in image.header.items():
            print(f"  {key} = {val}")

        
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


        # mk_l2_meta
        mk_l2_meta = {
            "l2_cal_step_match": {
                "assign_wcs": self.wcs_from_fits_header,
                "dark": self.dark_current,
                "flux": base['flux'],
                "linearity": self.nonlinearity,
            },
            "photometry_match": {},
            "outlier_detection_match": {},
            "sky_background_match": {
                "subtracted": self.sky_subtract,
            },
            "source_catalog_match": {},
            "cal_logs_match": {},
        }

        # mk_common_meta
        mk_common_meta = {
            "coordinates_match": {},
            "ephemeris_match": {
                "time": self.mjd
            },
            "exposure_match": {
                "type": config['type'],
                "start_time": image.header['DATE-OBS'],
                "exposure_time": image.header['EXPTIME'],
            },
            "guidestar_match": {},
            "wfi_mode_match": {
                "detector": self.sca,
                "optical_element": image.header['FILTER'],
            },
            "observation_match": {},
            #not sure if they need to be in deg or rad, just put in the original unit in image
            "pointing_match": {
                "target_ra": base['world_center'].ra,
                "target_dec": base['world_center'].dec,
            },
            "program_match": {},
            "rcs_match": {},
            "ref_file_match": {},
            "velocity_aberration_match": {
                # I don't think we have velocity aberration yet in sim
                #"ra_reference": base['world_center'].ra,
                #"dec_reference": base['world_center'].dec,
            },
            "visit_match": {
                #This is higher level when multiple exposure in same visit
                #"start_time": image.header['DATE-OBS'],
            },
            "wcsinfo_match": {
                "ra_ref": base['world_center'].ra,
                "dec_ref": base['world_center'].dec,
            },
        }

        # mk_basic_meta
        filename = path
        #the helper function only write on existing file. So create the asdf file if it is not there
        filepath = Path(filename)
        if not filepath.exists():
            filepath.touch()
            
        mod_time = os.path.getmtime(filename)
        file_date = Time(mod_time, format="unix").isot
        origin = "Duke"
        telescope = "ROMAN"

        mk_basic_meta = {
            "calibration_software_name_match": {},
            "calibration_software_version_match": {},
            "product_type_match": {},
            "filename_match": {"filename": filename},
            "file_date_match": {"file_date": file_date},
            "model_type_match": {},
            "origin_match": {"origin": origin},
            "prd_version_match": {},
            "sdf_software_version_match": {},
            "telescope_match": {"telescope": telescope},
        }

        # merge into one dictionary for meta
        meta_dict = {}
        meta_dict.update(mk_l2_meta)
        meta_dict.update(mk_common_meta)
        meta_dict.update(mk_basic_meta)


        # Fill out the wcs block - this is very hard since we need to change from wcs to gwcs object stored in asdf
        # The config wcs is the configuration for building the wcs above
        # The base is galsim.GSFitsWCS constructed from config wcs using wcs.py, and copied to image.wcs
        
        #turn the Galsim header to a fits header for the wcs_from_fits_header function
        galsim_wcs_header = image.wcs.header           
        d = dict(galsim_wcs_header)                     
        wcs_header = Header(d)
        ny, nx = image.array.shape
        
        wcs_header['NAXIS']  = 2                   # number of axes
        wcs_header['NAXIS1'] = nx                  # image width in pixels
        wcs_header['NAXIS2'] = ny                  # image height in pixels

        # coordinate type
        wcs_header['CTYPE1'] = 'RA---TAN-SIP'
        wcs_header['CTYPE2'] = 'DEC--TAN-SIP'

        # reference pixel 
        wcs_header['CRPIX1'] = nx/2
        wcs_header['CRPIX2'] = ny/2

        # reference coordinates 
        wcs_header['CRVAL1'] = base['world_center'].ra.deg   
        wcs_header['CRVAL2'] = base['world_center'].dec.deg    

        #I can't find these information (I think they are also connected to border_ref_pix_left etc) so comment below out - the wcs is not correct without these info
        # linear transformation: pixel scale in deg/pixel
        #scale = 0.1 / 3600.0          
        #if the image has rotation wrt constant ra and dec
        #wcs_header['CD1_1'] = -scale  
        #wcs_header['CD1_2'] = 0.0
        #wcs_header['CD2_1'] = 0.0
        #wcs_header['CD2_2'] = scale

        # coordinate system
        wcs_header['RADESYS'] = 'ICRS'
        wcs_header['LONPOLE'] = 180.0
        
        # wfi_image = mk_level2_image_with_wcs(
        #     shape=image.array.shape,
        #     filepath=path,
        #     data=image.array.astype('float32'),  #For now convert to float32, latter the image.array shall it self be float32
        #     meta=meta_dict,
        #     wcs = self.wcs_from_fits_header(wcs_header)
        # )
        
      
        return None

    def writeASDF_rdm(self, config, base, image, path, include_raw_header=False):
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

        # Fill out the wcs block - this is very hard since we need to change from wcs to gwcs object stored in asdf
        # The config wcs is the configuration for building the wcs above
        # The base is galsim.GSFitsWCS constructed from config wcs using wcs.py, and copied to image.wcs
        
        #turn the Galsim header to a fits header for the wcs_from_fits_header function
        galsim_wcs_header = image.wcs.header           
        d = dict(galsim_wcs_header)                     
        wcs_header = Header(d)
        ny, nx = image.array.shape
        
        wcs_header['NAXIS']  = 2                   # number of axes
        wcs_header['NAXIS1'] = nx                  # image width in pixels
        wcs_header['NAXIS2'] = ny                  # image height in pixels

        # coordinate type
        wcs_header['CTYPE1'] = 'RA---TAN-SIP'
        wcs_header['CTYPE2'] = 'DEC--TAN-SIP'

        # reference pixel 
        wcs_header['CRPIX1'] = nx/2
        wcs_header['CRPIX2'] = ny/2

        # reference coordinates 
        wcs_header['CRVAL1'] = base['world_center'].ra.deg   
        wcs_header['CRVAL2'] = base['world_center'].dec.deg    

        #I can't find these information (I think they are also connected to border_ref_pix_left etc) so comment below out - the wcs is not correct without these info
        # linear transformation: pixel scale in deg/pixel
        #scale = 0.1 / 3600.0          
        #if the image has rotation wrt constant ra and dec
        #wcs_header['CD1_1'] = -scale  
        #wcs_header['CD1_2'] = 0.0
        #wcs_header['CD2_1'] = 0.0
        #wcs_header['CD2_2'] = scale

        # coordinate system
        wcs_header['RADESYS'] = 'ICRS'
        wcs_header['LONPOLE'] = 180.0
        breakpoint()
        tree = ImageModel.create_fake_data()
        
        # setup value assignment in the same order as they appear in template
        # assigning default values: when attribute not understood or not available in the scope of the function.
        tree.meta.calibration_software_name = tree.meta.calibration_software_name
        tree.meta.calibration_software_version = tree.meta.calibration_software_version

        tree.meta.product_type = tree.meta.product_type

        tree.meta.filename = path
        tree.meta.file_date = tree.meta.file_date
        
        tree.meta.model_type = tree.meta.model_type
        
        tree.meta.origin = tree.meta.origin
        
        tree.meta.prd_version = tree.meta.prd_version
        
        tree.meta.sdf_software_version = tree.meta.sdf_software_version
        
        tree.meta.telescope = tree.meta.telescope
        
        tree.meta.coordinates.reference_frame = tree.meta.coordinates.reference_frame

        # tree["meta"]['file_date'] = #should automatically be set
        # tree["meta"]['model_type'] = 'ImageModel'
        # tree["meta"]['origin'] = 'STSCI/SOC'
        # tree["meta"]['prd_version'] = '?'
        # tree["meta"]['sdf_software_version'] = '?'
        tree["meta"]['telescope'] = 'ROMAN'
        # tree["meta"]["coordinates"] = {'reference_frame': 'ICRS'}
        # tree["meta"]["ephemeris"] = {
        # 'ephemeris_reference_frame': '?',
        # 'type': 'DEFINITIVE', 
        # 'time': -999999.0, 
        # 'spatial_x': -999999.0, 
        # 'spatial_y': -999999.0, 
        # 'spatial_z': -999999.0, 
        # 'velocity_x': -999999.0, 
        # 'velocity_y': -999999.0, 
        # 'velocity_z': -999999.0
        # }
        # tree["meta"]['exposure'] = {
        # 'type': 'WFI_IMAGE', 
        # 'start_time': <Time object: scale='utc' format='isot' value=2020-01-01T00:00:00.000>, 
        # 'end_time': <Time object: scale='utc' format='isot' value=2020-01-01T00:00:00.000>, 
        # 'engineering_quality': 'OK', 
        # 'ma_table_id': '?', 
        # 'nresultants': -999999, 
        # 'data_problem': '?', 
        # 'frame_time': -999999.0, 
        # 'exposure_time': -999999.0, 
        # 'effective_exposure_time': -999999.0, 
        # 'ma_table_name': '?', 
        # 'ma_table_number': -999999, 
        # 'read_pattern': [], 
        # 'truncated': False
        # }
        
        #tree["meta"]['guide_star'] = {
        #    'guide_window_id': '?', 
        #    'guide_mode': 'WIM-ACQ', 
        #    'window_xstart': -999999, 
        #    'window_ystart': -999999, 
        #    'window_xstop': -999999, 
        #    'window_ystop': -999999, 
        #    'guide_star_id': '?', 
        #    'epoch': '?'
        #    }
        
        tree["meta"]['instrument'] = {
            'name': 'WFI', 
            'detector': 'WFI01', 
            'optical_element': params["filter"]
            }
        
        tree["meta"]['observation'] = {
            'observation_id': '?', 
            'visit_id': '?', 
            'program': -999999, 
            'execution_plan': -999999, 
            'pass': -999999, 
            'segment': -999999, 
            'observation': -999999, 
            'visit': base['input']['obseq_data']['visit'], 
            'visit_file_group': -999999, 
            'visit_file_sequence': -999999, 
            'visit_file_activity': '?', 
            'exposure': image.header['EXPTIME'], 
            'wfi_parallel': False
            }
        
        tree["meta"]['pointing'] = {
            'pa_aperture': -999999.0, 
            'pointing_engineering_source': 'CALCULATED', 
            'ra_v1': -999999.0, 
            'dec_v1': -999999.0, 
            'pa_v3': -999999.0, 
            'target_aperture': 'WFI_CEN', 
            'target_ra': image.wcs.center.ra.deg, 
            'target_dec': image.wcs.center.dec.deg
            }
  
        # tree["meta"]['program'] = {
        # 'title': '?',
        # 'investigator_name': '?',
        # 'category': '?',
        # 'subcategory': 'CAL',
        # 'science_category': '?'
        # }
        # tree["meta"]['ref_file'] = {
        # {'crds': {'version': '?', 'context': '?'}, 
        # 'apcorr': '?', 
        # 'area': '?', 
        # 'dark': '?', 
        # 'darkdecaysignal': '?', 
        # 'distortion': '?', 
        # 'epsf': '?', 
        # 'mask': '?', 
        # 'flat': '?', 
        # 'gain': '?', 
        # 'inverselinearity': '?', 
        # 'linearity': '?', 
        # 'integralnonlinearity': '?', 
        # 'photom': '?', 
        # 'readnoise': '?', 
        # 'refpix': '?', 
        # 'saturation': '?'
        # }
        # tree["meta"]['rcs'] = {'active': False, 'electronics': 'A', 'bank': '1', 'led': '1', 'counts': -999999}
        # tree["meta"]['velocity_aberration'] = {'ra_reference': -999999.0, 'dec_reference': -999999.0, 'scale_factor': -999999.0}



        
        
        tree["meta"]['instrument']['detector'] = 'WFI10'
        tree["meta"]['instrument']['optical_element'] = self.filter
        tree["meta"]['instrument']['name'] = 'WFI'

        tree["meta"]['obs_date'] = Time(self.mjd, format="mjd").datetime.isoformat()
        tree["meta"]['pointing'] = {
            'pa_aperture': -999999.0, 
            'pointing_engineering_source': 'CALCULATED', 
            'ra_v1': -999999.0, 
            'dec_v1': -999999.0, 
            'pa_v3': -999999.0, 
            'target_aperture': 'WFI_CEN', 
            'target_ra': image.wcs.center.ra.deg, 
            'target_dec': image.wcs.center.dec.deg
            }
        tree["meta"]['exposure_time'] = self.exptime
        tree["meta"]['mjd_obs'] = self.mjd
        tree["meta"]['nreads'] = 1
        tree["meta"]['gain'] = 1.0
        #tree["meta"]['sky_mean'] = 0.0  # Placeholder for sky mean, as it's not currently implemented in the yaml
        tree["meta"]['zptmag'] = image.header['ZPTMAG']    #2.5 * np.log10(self.exptime * roman.collecting_area)
        tree["meta"]['pa_fpa'] = True

        
        
        tree.meta.ephemeris.ephemeris_reference_frame = tree.meta.ephemeris.ephemeris_reference_frame
        tree.meta.ephemeris.type = tree.meta.ephemeris.type
        tree.meta.ephemeris.time = tree.meta.ephemeris.time
        tree.meta.ephemeris.spatial_x = tree.meta.ephemeris.spatial_x
        tree.meta.ephemeris.spatial_y = tree.meta.ephemeris.spatial_y
        tree.meta.ephemeris.spatial_z = tree.meta.ephemeris.spatial_z
        tree.meta.ephemeris.velocity_x = tree.meta.ephemeris.velocity_x
        tree.meta.ephemeris.velocity_y = tree.meta.ephemeris.velocity_y
        tree.meta.ephemeris.velocity_z = tree.meta.ephemeris.velocity_z
        
        tree.meta.exposure.type = tree.meta.exposure.type
        tree.meta.exposure.start_time = tree.meta.exposure.start_time
        tree.meta.exposure.end_time = tree.meta.exposure.end_time
        tree.meta.exposure.engineering_quality = tree.meta.exposure.engineering_quality
        tree.meta.exposure.ma_table_id = tree.meta.exposure.ma_table_id
        tree.meta.exposure.nresultants = tree.meta.exposure.nresultants
        tree.meta.exposure.data_problem = tree.meta.exposure.data_problem
        tree.meta.exposure.frame_time = tree.meta.exposure.frame_time
        tree.meta.exposure.exposure_time = tree.meta.exposure.exposure_time
        tree.meta.exposure.effective_exposure_time = tree.meta.exposure.effective_exposure_time
        tree.meta.exposure.ma_table_name = tree.meta.exposure.ma_table_name
        tree.meta.exposure.ma_table_number = tree.meta.exposure.ma_table_number
        tree.meta.exposure.truncated = tree.meta.exposure.truncated
        
        tree.meta.guide_star.guide_window_id = tree.meta.guide_star.guide_window_id
        tree.meta.guide_star.guide_mode = tree.meta.guide_star.guide_mode
        tree.meta.guide_star.window_xstart = tree.meta.guide_star.window_xstart
        tree.meta.guide_star.window_ystart = tree.meta.guide_star.window_ystart
        tree.meta.guide_star.window_xstop = tree.meta.guide_star.window_xstop
        tree.meta.guide_star.window_ystop = tree.meta.guide_star.window_ystop
        tree.meta.guide_star.guide_star_id = tree.meta.guide_star.guide_star_id
        tree.meta.guide_star.epoch = tree.meta.guide_star.epoch
        
        tree.meta.instrument.name = tree.meta.instrument.name
        tree.meta.instrument.detector = tree.meta.instrument.detector
        tree.meta.instrument.optical_element = tree.meta.instrument.optical_element
        
        tree.meta.observation.observation_id = tree.meta.observation.observation_id
        tree.meta.observation.visit_id = tree.meta.observation.visit_id
        tree.meta.observation.program = tree.meta.observation.program
        tree.meta.observation.execution_plan = tree.meta.observation.execution_plan
        tree.meta.observation.pass = tree.meta.observation.pass
        tree.meta.observation.segment = tree.meta.observation.segment
        tree.meta.observation.observation = tree.meta.observation.observation
        tree.meta.observation.visit = tree.meta.observation.visit
        tree.meta.observation.visit_file_group = tree.meta.observation.visit_file_group
        tree.meta.observation.visit_file_sequence = tree.meta.observation.visit_file_sequence
        tree.meta.observation.visit_file_activity = tree.meta.observation.visit_file_activity
        tree.meta.observation.exposure = tree.meta.observation.exposure
        tree.meta.observation.wfi_parallel = tree.meta.observation.wfi_parallel
        
        tree.meta.pointing.pa_aperture = tree.meta.pointing.pa_aperture
        tree.meta.pointing.pointing_engineering_source = tree.meta.pointing.pointing_engineering_source
        tree.meta.pointing.ra_v1 = tree.meta.pointing.ra_v1
        tree.meta.pointing.dec_v1 = tree.meta.pointing.dec_v1
        tree.meta.pointing.pa_v3 = tree.meta.pointing.pa_v3
        tree.meta.pointing.target_aperture = tree.meta.pointing.target_aperture
        tree.meta.pointing.target_ra = tree.meta.pointing.target_ra
        tree.meta.pointing.target_dec = tree.meta.pointing.target_dec
        tree.meta.program.title = tree.meta.program.title
        tree.meta.program.investigator_name = tree.meta.program.investigator_name
        tree.meta.program.category = tree.meta.program.category
        tree.meta.program.subcategory = tree.meta.program.subcategory
        tree.meta.program.science_category = tree.meta.program.science_category
        
        tree.meta.ref_file.crds.version = tree.meta.ref_file.crds.version
        tree.meta.ref_file.crds.context = tree.meta.ref_file.crds.context
        tree.meta.ref_file.apcorr = tree.meta.ref_file.apcorr
        tree.meta.ref_file.area = tree.meta.ref_file.area
        tree.meta.ref_file.dark = tree.meta.ref_file.dark
        tree.meta.ref_file.darkdecaysignal = tree.meta.ref_file.darkdecaysignal
        tree.meta.ref_file.distortion = tree.meta.ref_file.distortion
        tree.meta.ref_file.epsf = tree.meta.ref_file.epsf
        tree.meta.ref_file.mask = tree.meta.ref_file.mask
        tree.meta.ref_file.flat = tree.meta.ref_file.flat
        tree.meta.ref_file.gain = tree.meta.ref_file.gain
        tree.meta.ref_file.inverselinearity = tree.meta.ref_file.inverselinearity
        tree.meta.ref_file.linearity = tree.meta.ref_file.linearity
        tree.meta.ref_file.integralnonlinearity = tree.meta.ref_file.integralnonlinearity
        tree.meta.ref_file.photom = tree.meta.ref_file.photom
        tree.meta.ref_file.readnoise = tree.meta.ref_file.readnoise
        tree.meta.ref_file.refpix = tree.meta.ref_file.refpix
        tree.meta.ref_file.saturation = tree.meta.ref_file.saturation
        
        tree.meta.rcs.active = tree.meta.rcs.active
        tree.meta.rcs.electronics = tree.meta.rcs.electronics
        tree.meta.rcs.bank = tree.meta.rcs.bank
        tree.meta.rcs.led = tree.meta.rcs.led
        tree.meta.rcs.counts = tree.meta.rcs.counts
        
        tree.meta.velocity_aberration.ra_reference = tree.meta.velocity_aberration.ra_reference
        tree.meta.velocity_aberration.dec_reference = tree.meta.velocity_aberration.dec_reference
        tree.meta.velocity_aberration.scale_factor = tree.meta.velocity_aberration.scale_factor
        
        #meta.visit
        tree.meta.visit.dither.primary_name  = tree.meta.wcsinfo.dither.primary_name
        tree.meta.visit.dither.subpixel_name = tree.meta.wcsinfo.dither.subpixel_name
        tree.meta.visit.type                 = tree.meta.wcsinfo.type
        tree.meta.visit.start_time           = tree.meta.wcsinfo.start_time
        tree.meta.visit.nexposures           = tree.meta.wcsinfo.nexposures
        tree.meta.visit.internal_target      = tree.meta.wcsinfo.internal_target
        #meta.wcs
        tree.meta.wcs = self.wcs_from_fits_header(wcs_header)
        #meta.wcsinfo
        tree.meta.wcsinfo.aperture_name = tree.meta.wcsinfo.aperture_name
        tree.meta.wcsinfo.v2_ref        = tree.meta.wcsinfo.v2_ref
        tree.meta.wcsinfo.v3_ref        = tree.meta.wcsinfo.v3_ref
        tree.meta.wcsinfo.vparity       = tree.meta.wcsinfo.vparity
        tree.meta.wcsinfo.v3yangle      = tree.meta.wcsinfo.v3yangle
        tree.meta.wcsinfo.ra_ref        = tree.meta.wcsinfo.ra_ref
        tree.meta.wcsinfo.dec_ref       = tree.meta.wcsinfo.dec_ref
        tree.meta.wcsinfo.roll_ref      = tree.meta.wcsinfo.roll_ref
        tree.meta.wcsinfo.s_region      = tree.meta.wcsinfo.s_region
        #meta.photometry
        tree.meta.photometry.conversion_megajanskys = tree.meta.photometry.conversion_megajanskys
        tree.meta.photometry.conversion_megajanskys_uncertainty = tree.meta.photometry.conversion_megajanskys_uncertainty
        tree.meta.photometry.pixel_area = tree.meta.photometry.pixel_area

        tree['data'] = image.array.astype('float32')
        tree["err"] = np.zeros_like(image.array, dtype='float32')  # Placeholder for error array
        tree["dq"] = np.zeros_like(image.array, dtype='uint32')  # Placeholder for data quality array

        

        _ = tree.save(path, dir_path=None)
      

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
        breakpoint()
        print("full image wcs:", full_image.wcs)
        print("full image wcs type:", type(full_image.wcs))
        print("full image wcs header:", full_image.wcs.header)
        print("full image wcs header type:", type(full_image.wcs.header))
        print("full image wcs header keys:", full_image.wcs.header.keys())
        print("full image wcs methods:", dir(full_image.wcs))

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
        for batch in range(3):#range(nbatch):
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
            self.writeASDF_rdm(config, base, full_image, 'one_image.asdf', include_raw_header=False)
        
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
