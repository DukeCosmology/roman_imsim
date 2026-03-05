from pathlib import Path

import astropy.coordinates
import astropy.units as u
import galsim
import gwcs
import numpy as np
from astropy import wcs as fits_wcs
from astropy.modeling.models import (
    Mapping,
    Pix2Sky_TAN,
    Polynomial2D,
    RotateNative2Celestial,
    Shift,
)
from galsim.config.output import OutputBuilder, RegisterOutputType
from gwcs import coordinate_frames as cf
from roman_datamodels.datamodels import ImageModel

__all__ = ["RomanASDFBuilder"]


class RomanASDFBuilder(OutputBuilder):
    """Builder class for constructing and writing DataCube output types."""

    # override the base class variable
    default_ext = ".asdf"

    def buildImages(self, config, base, file_num, image_num, obj_num, ignore, logger):
        """Build the images for output.

        In the base class, this function just calls BuildImage to build the single image to
        put in the output file.  So the returned list only has one item.

        Parameters:
            config:     The configuration dict for the output field.
            base:       The base configuration dict.
            file_num:   The current file_num.
            image_num:  The current image_num.
            obj_num:    The current obj_num.
            ignore:     A list of parameters that are allowed to be in config that we can
                        ignore here.  i.e. it won't be an error if they are present.
            logger:     If given, a logger object to log progress.

        Returns:
            a list of the images built
        """
        # There are no extra parameters to get, so just check that there are no invalid parameters
        # in the config dict.
        opt = {
            "include_raw_header": bool,
        }
        ignore += ["file_name", "dir", "nfiles"]
        params = galsim.config.GetAllParams(config, base, opt=opt, ignore=ignore)[0]
        self.include_raw_header = params.get("include_raw_header", False)

        image = galsim.config.BuildImage(base, image_num, obj_num, logger=logger)
        return [image]

    def writeFile(self, data, file_name, config, base, logger):
        """Write the data to a file.

        Parameters:
            data:           The data to write.  Usually a list of images returned by
                            buildImages, but possibly with extra HDUs tacked onto the end
                            from the extra output items.
            file_name:      The file_name to write to.
            config:         The configuration dict for the output field.
            base:           The base configuration dict.
            logger:         If given, a logger object to log progress.
        """
        if Path(file_name).suffix != ".asdf":
            raise NotImplementedError(
                f"The extension of the file_name/format MUST be {self.default_ext}. "
                "Roman_datamodels only allows asdf and parquet."
            )

        self._writeASDF(config, base, data[0], file_name, logger)

    @staticmethod
    def wcs_from_fits_header(header):
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
                        setattr(pol, f"c{i}_{j}", mat[i, j])
            return pol

        w = fits_wcs.WCS(header)
        ny, nx = header["NAXIS2"] + 1, header["NAXIS1"] + 1
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
            (Shift(-x0) & Shift(-y0))
            | Mapping((0, 1, 0, 1))
            | (polx & poly)
            | Pix2Sky_TAN()
            | RotateNative2Celestial(*w.wcs.crval, header["LONPOLE"])
        )

        detector_frame = cf.Frame2D(name="detector", axes_names=("x", "y"), unit=(u.pix, u.pix))
        sky_frame = cf.CelestialFrame(
            reference_frame=getattr(astropy.coordinates, w.wcs.radesys).__call__(),
            name=w.wcs.radesys,
            unit=(u.deg, u.deg),
        )
        pipeline = [(detector_frame, det2sky), (sky_frame, None)]
        gw = gwcs.WCS(pipeline)
        gw.bounding_box = ((-0.5, nx - 0.5), (-0.5, ny - 0.5))

        return gw

    def _writeASDF(self, config, base, image, fname_path, logger):
        """
        Method to write the file to disk

        Parameters:
        -----------
        fname_path (str): Output path and filename
        include_raw_header (bool): If `True` include a copy of the raw FITS header
           as a dictionary in the ASDF file.
        """

        sca = image.header["SCA"]
        exptime = image.header["EXPTIME"]
        fltr = image.header["FILTER"]
        date_obs = image.header["DATE-OBS"]
        mjd_obs = image.header["MJD-OBS"]
        ZPTMAG = image.header["ZPTMAG"]

        # Fill out the wcs block - this is very hard since we need to change
        # from wcs to gwcs object stored in asdf
        # The config wcs is the configuration for building the wcs above
        # The base is galsim.GSFitsWCS constructed from config wcs using wcs.py
        # and copied to image.wcs

        # use astropy.io.fits.header.Header form of header: required in wcs_from_fits_header
        wcs_header = image.wcs.header.header
        # Don't remove, these are needed to create gwcs object.
        ny, nx = image.array.shape
        wcs_header["NAXIS"] = 2  # number of axes
        wcs_header["NAXIS1"] = nx  # image width in pixels
        wcs_header["NAXIS2"] = ny  # image height in pixels

        # already assigned, redundant repetition?
        # coordinate type
        # wcs_header['CTYPE1'] = 'RA---TAN-SIP'
        # wcs_header['CTYPE2'] = 'DEC--TAN-SIP'

        # already assigned, redundant repetition, but conversion to int is required
        # comment/delete this block if float is fine. This too is a repetition again.
        # reference pixel
        # wcs_header['CRPIX1'] = nx/2
        # wcs_header['CRPIX2'] = ny/2
        # Is the repetition really needed? already defined! but default value differs from assigned.
        # reference coordinates
        # wcs_header['CRVAL1'] = base['world_center'].ra.deg
        # wcs_header['CRVAL2'] = base['world_center'].dec.deg

        # Agreed with Arun to remove this part. Keeping commented for now.
        # I can't find these information
        # (I think they are also connected to border_ref_pix_left etc) so comment below out
        # - the wcs is not correct without these info
        # linear transformation: pixel scale in deg/pixel
        # scale = 0.1 / 3600.0
        # if the image has rotation wrt constant ra and dec
        # wcs_header['CD1_1'] = -scale
        # wcs_header['CD1_2'] = 0.0
        # wcs_header['CD2_1'] = 0.0
        # wcs_header['CD2_2'] = scale

        # coordinate system
        wcs_header.update([("RADESYS", "ICRS", "Reference Coordinate System")])
        # also already defined
        # wcs_header['LONPOLE'] = 180.0

        # changing the instrument name from WFC -> WFI, OK?
        wcs_header["INSTRUME"] = "WFI"

        tree = ImageModel.create_fake_data()

        # Put additional attributes (that do NOT exist in Roman_datamodels) in this block
        if self.include_raw_header:
            tree["meta"]["raw_wcs_header"] = {}
            for card in wcs_header.cards:
                logger.info("keyword: %s, value: %s, comment: %s", card.keyword, card.value, card.comment)
                tree["meta"]["raw_wcs_header"][card.keyword] = {
                    "value": card.value,
                    "comment": card.comment,
                }
        # save() call autometically creates a file_date, so you don't have to
        # pass a file creation/modification date. Here obs_date is the observation date.
        tree["meta"]["date_obs"] = {"value": date_obs, "comment": "observation date"}
        tree["meta"]["mjd_obs"] = {
            "value": mjd_obs,
            "comment": "obsevation date in mjd",
        }
        tree["meta"]["nreads"] = {"value": 1, "comment": ""}
        tree["meta"]["gain"] = {"value": 1.0, "comment": ""}
        # tree["meta"]['sky_mean'] = 0.0 # Placeholder for sky mean, as it's not currently implemented
        tree["meta"]["zptmag"] = {
            "value": ZPTMAG,
            "comment": (
                "Instrumental zero-point magnitude to get the apparaent magnitude that is independent of "
                "collecting area and exposure time. Calculated as "
                "- 2.5 *np.log10*(flux) + np.log10(exptime *collecting_area)"
            ),
        }
        tree["meta"]["pa_fpa"] = True

        # ------------------------------------------
        # setup value assignment in the same order as they appear in RDM
        # template. Assigning default values when attribute not understood or
        # not available in the scope of the function.
        # --------------------------------------------------
        tree.meta.calibration_software_name = tree.meta.calibration_software_name
        tree.meta.calibration_software_version = tree.meta.calibration_software_version

        tree.meta.product_type = tree.meta.product_type

        tree.meta.filename = Path(fname_path).name  # ===> save() appies this already. confirm!

        # tree.meta.file_date #=> don't assign value, it's set autometically at save() call

        tree.meta.model_type = tree.meta.model_type  # 'ImageModel'

        tree.meta.origin = tree.meta.origin  # defaults to 'STSCI/SOC'

        tree.meta.prd_version = tree.meta.prd_version

        tree.meta.sdf_software_version = tree.meta.sdf_software_version

        tree.meta.telescope = tree.meta.telescope

        tree.meta.coordinates.reference_frame = tree.meta.coordinates.reference_frame
        # ===> I'm not sure this is the correct way to assigning.
        # check by saving the file and checking the data type of tree["meta"]["coordinates"]
        # tree["meta"]["coordinates"] = {'reference_frame': 'ICRS'}

        # meta.ephemeris
        tree.meta.ephemeris.ephemeris_reference_frame = tree.meta.ephemeris.ephemeris_reference_frame
        tree.meta.ephemeris.type = tree.meta.ephemeris.type
        tree.meta.ephemeris.time = tree.meta.ephemeris.time
        tree.meta.ephemeris.spatial_x = tree.meta.ephemeris.spatial_x
        tree.meta.ephemeris.spatial_y = tree.meta.ephemeris.spatial_y
        tree.meta.ephemeris.spatial_z = tree.meta.ephemeris.spatial_z
        tree.meta.ephemeris.velocity_x = tree.meta.ephemeris.velocity_x
        tree.meta.ephemeris.velocity_y = tree.meta.ephemeris.velocity_y
        tree.meta.ephemeris.velocity_z = tree.meta.ephemeris.velocity_z
        # .meta.exposure
        tree.meta.exposure.type = tree.meta.exposure.type
        tree.meta.exposure.start_time = tree.meta.exposure.start_time
        tree.meta.exposure.end_time = tree.meta.exposure.end_time
        tree.meta.exposure.engineering_quality = tree.meta.exposure.engineering_quality
        tree.meta.exposure.ma_table_id = tree.meta.exposure.ma_table_id
        tree.meta.exposure.nresultants = tree.meta.exposure.nresultants
        tree.meta.exposure.data_problem = tree.meta.exposure.data_problem
        tree.meta.exposure.frame_time = tree.meta.exposure.frame_time
        tree.meta.exposure.exposure_time = exptime
        tree.meta.exposure.effective_exposure_time = tree.meta.exposure.effective_exposure_time
        tree.meta.exposure.ma_table_name = tree.meta.exposure.ma_table_name
        tree.meta.exposure.ma_table_number = tree.meta.exposure.ma_table_number
        tree.meta.exposure.truncated = tree.meta.exposure.truncated
        # meta.guide_star
        tree.meta.guide_star.guide_window_id = tree.meta.guide_star.guide_window_id
        tree.meta.guide_star.guide_mode = tree.meta.guide_star.guide_mode
        tree.meta.guide_star.window_xstart = tree.meta.guide_star.window_xstart
        tree.meta.guide_star.window_ystart = tree.meta.guide_star.window_ystart
        tree.meta.guide_star.window_xstop = tree.meta.guide_star.window_xstop
        tree.meta.guide_star.window_ystop = tree.meta.guide_star.window_ystop
        tree.meta.guide_star.guide_star_id = tree.meta.guide_star.guide_star_id
        tree.meta.guide_star.epoch = tree.meta.guide_star.epoch
        # meta.instrument
        tree.meta.instrument.name = wcs_header["INSTRUME"]
        tree.meta.instrument.detector = f"{wcs_header['INSTRUME']}{sca:02}"
        tree.meta.instrument.optical_element = "F" + fltr[1:]
        # mata.observation
        tree.meta.observation.observation_id = tree.meta.observation.observation_id
        tree.meta.observation.visit_id = tree.meta.observation.visit_id
        tree.meta.observation.program = tree.meta.observation.program
        tree.meta.observation.execution_plan = tree.meta.observation.execution_plan
        # pass being a special python-statement, dot call on attr named 'pass' ends up as Error
        # tree.meta.observation["pass'] = tree.meta.observation["pass"]
        tree.meta.observation.segment = tree.meta.observation.segment
        tree.meta.observation.observation = tree.meta.observation.observation
        tree.meta.observation.visit = base["input"]["obseq_data"]["visit"]
        tree.meta.observation.visit_file_group = tree.meta.observation.visit_file_group
        tree.meta.observation.visit_file_sequence = tree.meta.observation.visit_file_sequence
        tree.meta.observation.visit_file_activity = tree.meta.observation.visit_file_activity
        # The exposure number of the sequence (visit).
        # If you do 2 dithers the first one is exp 1, and the second is exp 2.
        tree.meta.observation.exposure = tree.meta.observation.exposure
        tree.meta.observation.wfi_parallel = tree.meta.observation.wfi_parallel
        # meta.pointing
        tree.meta.pointing.pa_aperture = tree.meta.pointing.pa_aperture
        tree.meta.pointing.pointing_engineering_source = tree.meta.pointing.pointing_engineering_source
        tree.meta.pointing.ra_v1 = tree.meta.pointing.ra_v1
        tree.meta.pointing.dec_v1 = tree.meta.pointing.dec_v1
        tree.meta.pointing.pa_v3 = tree.meta.pointing.pa_v3
        tree.meta.pointing.target_aperture = (
            f"{wcs_header['INSTRUME']}_CEN"  # what kind of aperture is wcs_header['RA_TARG']
        )
        tree.meta.pointing.target_ra = image.wcs.center.ra.deg  # or wcs_header['RA_TARG']?
        tree.meta.pointing.target_dec = image.wcs.center.dec.deg  # or wcs_header['DEC_TARG']?
        # meta.program
        tree.meta.program.title = tree.meta.program.title
        tree.meta.program.investigator_name = tree.meta.program.investigator_name
        tree.meta.program.category = tree.meta.program.category
        tree.meta.program.subcategory = tree.meta.program.subcategory
        tree.meta.program.science_category = tree.meta.program.science_category
        # meta.ref_file
        tree.meta.ref_file.crds.version = tree.meta.ref_file.crds.version
        tree.meta.ref_file.crds.context = tree.meta.ref_file.crds.context
        tree.meta.ref_file.apcorr = tree.meta.ref_file.apcorr
        tree.meta.ref_file.area = tree.meta.ref_file.area
        tree.meta.ref_file.dark = tree.meta.ref_file.dark
        # tree.meta.ref_file.darkdecaysignal = tree.meta.ref_file.darkdecaysignal
        tree.meta.ref_file.distortion = tree.meta.ref_file.distortion
        tree.meta.ref_file.epsf = tree.meta.ref_file.epsf
        tree.meta.ref_file.mask = tree.meta.ref_file.mask
        tree.meta.ref_file.flat = tree.meta.ref_file.flat
        tree.meta.ref_file.gain = tree.meta.ref_file.gain
        # tree.meta.ref_file.inverselinearity = tree.meta.ref_file.inverselinearity
        tree.meta.ref_file.linearity = tree.meta.ref_file.linearity
        # tree.meta.ref_file.integralnonlinearity = tree.meta.ref_file.integralnonlinearity
        tree.meta.ref_file.photom = tree.meta.ref_file.photom
        tree.meta.ref_file.readnoise = tree.meta.ref_file.readnoise
        tree.meta.ref_file.refpix = tree.meta.ref_file.refpix
        tree.meta.ref_file.saturation = tree.meta.ref_file.saturation
        # .meta.rcs
        tree.meta.rcs.active = tree.meta.rcs.active
        tree.meta.rcs.electronics = tree.meta.rcs.electronics
        tree.meta.rcs.bank = tree.meta.rcs.bank
        tree.meta.rcs.led = tree.meta.rcs.led
        tree.meta.rcs.counts = tree.meta.rcs.counts
        # meta.velocity_aberration
        tree.meta.velocity_aberration.ra_reference = tree.meta.velocity_aberration.ra_reference
        tree.meta.velocity_aberration.dec_reference = tree.meta.velocity_aberration.dec_reference
        tree.meta.velocity_aberration.scale_factor = tree.meta.velocity_aberration.scale_factor
        # meta.visit
        tree.meta.visit.dither.primary_name = tree.meta.visit.dither.primary_name
        tree.meta.visit.dither.subpixel_name = tree.meta.visit.dither.subpixel_name
        tree.meta.visit.type = tree.meta.visit.type
        tree.meta.visit.start_time = tree.meta.visit.start_time
        tree.meta.visit.nexposures = tree.meta.visit.nexposures
        tree.meta.visit.internal_target = tree.meta.visit.internal_target
        # meta.wcs
        tree.meta.wcs = self.wcs_from_fits_header(wcs_header)
        # meta.wcsinfo
        tree.meta.wcsinfo.aperture_name = (
            f"{wcs_header['INSTRUME']}{sca:02}_FULL"  # what does full stand for? # FULL or CEN?
        )
        tree.meta.wcsinfo.v2_ref = tree.meta.wcsinfo.v2_ref
        tree.meta.wcsinfo.v3_ref = tree.meta.wcsinfo.v3_ref
        tree.meta.wcsinfo.vparity = tree.meta.wcsinfo.vparity
        tree.meta.wcsinfo.v3yangle = tree.meta.wcsinfo.v3yangle
        tree.meta.wcsinfo.ra_ref = tree.meta.wcsinfo.ra_ref
        tree.meta.wcsinfo.dec_ref = tree.meta.wcsinfo.dec_ref
        tree.meta.wcsinfo.roll_ref = tree.meta.wcsinfo.roll_ref
        tree.meta.wcsinfo.s_region = tree.meta.wcsinfo.s_region
        # meta.photometry
        tree.meta.photometry.conversion_megajanskys = tree.meta.photometry.conversion_megajanskys
        tree.meta.photometry.conversion_megajanskys_uncertainty = (
            tree.meta.photometry.conversion_megajanskys_uncertainty
        )
        tree.meta.photometry.pixel_area = tree.meta.photometry.pixel_area

        tree.data = image.array.astype("float32")
        tree.dq = np.zeros_like(image.array, dtype="uint32")  # Placeholder for data quality array
        tree.err = np.zeros_like(image.array, dtype="float16")  # Placeholder for error array
        tree.var_poisson = tree.var_poisson
        tree.chisq = tree.chisq
        tree.dumo = tree.dumo
        tree.amp33 = tree.amp33
        tree.border_ref_pix_left = tree.border_ref_pix_left
        tree.border_ref_pix_right = tree.border_ref_pix_right
        tree.border_ref_pix_top = tree.border_ref_pix_top
        tree.border_ref_pix_bottom = tree.border_ref_pix_bottom
        tree.dq_border_ref_pix_left = tree.dq_border_ref_pix_left
        tree.dq_border_ref_pix_right = tree.dq_border_ref_pix_right
        tree.dq_border_ref_pix_top = tree.dq_border_ref_pix_top
        tree.dq_border_ref_pix_bottom = tree.dq_border_ref_pix_bottom

        _ = tree.to_asdf(fname_path)
        logger.info(f"saved {fname_path}")


RegisterOutputType("RomanASDF", RomanASDFBuilder())
