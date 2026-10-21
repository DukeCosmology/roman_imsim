from pathlib import Path
from copy import deepcopy
from collections.abc import Mapping as MappingABC

import asdf
from astropy.time import Time
from romanisim import l1
from romanisim import models

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
    """Write legacy 2-D images or CMOS L1 resultants in the Roman ASDF models."""

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
            "level": int,
            "data_encoding_offset": int,
        }
        ignore += ["file_name", "dir", "nfiles"]
        params = galsim.config.GetAllParams(config, base, opt=opt, ignore=ignore)[0]
        self.include_raw_header = params.get("include_raw_header", False)
        is_cmos = base['image'].get('type') == 'roman_sca_cmos'
        self.level = params.get('level', 1 if is_cmos else 2)
        self.data_encoding_offset = params.get('data_encoding_offset', None)
        if self.level not in (1, 2) or (self.level == 1) != is_cmos:
            raise galsim.GalSimConfigError('RomanASDF level 1 requires roman_sca_cmos; level 2 requires a 2-D image builder')
        base.pop('_cmos_ramp', None)

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

        if getattr(self, 'level', 2) == 1:
            self._writeL1(base, data[0], file_name, logger)
        else:
            self._writeASDF(config, base, data[0], file_name, logger)

    def _writeL1(self, base, image, file_name, logger):
        """Package already sampled CMOS resultants, without re-apportioning counts."""
        ramp = base.get('_cmos_ramp')
        if ramp is None:
            raise galsim.GalSimConfigError('CMOS builder did not produce an L1 ramp')
        expected = (models.parameters.n_pix, models.parameters.n_pix)
        if ramp.resultants.shape[1:] != expected:
            raise galsim.GalSimConfigError(f'L1 output requires science shape {expected}, got {ramp.resultants.shape[1:]}')
        metadata = deepcopy(models.parameters.default_parameters_dictionary)
        start = Time(image.header['MJD-OBS'], format='mjd')
        exposure = metadata.setdefault('exposure', {})
        exposure.update(start_time=start, mid_time=start + ramp.exposure_time / 2 * u.s,
                        end_time=start + ramp.exposure_time * u.s,
                        exposure_time=ramp.exposure_time, frame_time=ramp.frame_time,
                        effective_exposure_time=float(np.mean(ramp.read_pattern[-1])) * ramp.frame_time,
                        nresultants=len(ramp.read_pattern) + int(ramp.reference_read is not None),
                        read_pattern=ramp.read_pattern, type='WFI_IMAGE')
        instrument = metadata.setdefault('instrument', {})
        instrument.update(name='WFI', detector=f"WFI{image.header['SCA']:02d}",
                          optical_element='F' + image.header['FILTER'][1:])
        obseq = base.get('input', {}).get('obseq_data', {})
        if isinstance(obseq, dict) and 'visit' in obseq:
            metadata.setdefault('observation', {})['visit'] = galsim.config.ParseValue(obseq, 'visit', base, int)[0]
        if image.wcs.isCelestial():
            center = image.wcs.toWorld(image.true_center)
            metadata.setdefault('pointing', {}).update(target_ra=center.ra.deg, target_dec=center.dec.deg)
        # A custom pattern must not inherit an unrelated MA table identity.
        loader = galsim.config.GetInputObj('resultant_data', base['image'], base, 'RomanASDF')
        ma_table = loader.get('ma_table_number', 0)
        exposure['ma_table_number'] = ma_table
        exposure['ma_table_id'] = f'SCI{ma_table:04d}'
        raw, extras = l1.make_asdf(ramp.resultants, dq=ramp.dq,
                                  reference_read=ramp.reference_read,
                                  data_encoding_offset=self.data_encoding_offset)
        # make_asdf's shallow metadata update can replace required fields in
        # newer Roman schemas.  Merge into the schema-complete template instead.
        def merge(target, source):
            for key, value in source.items():
                if isinstance(value, MappingABC) and key in target and hasattr(target[key], 'items'):
                    merge(target[key], value)
                else:
                    target[key] = value
        merge(raw.meta, metadata)
        raw.meta.filename = Path(file_name).name
        extras.update(read_pattern=ramp.read_pattern, frame_time=ramp.frame_time,
                      gain=np.asarray(ramp.gain), source='roman_imsim CMOS photon sampling',
                      roman_imsim_version=image.header.get('VERSION', 'unknown'))
        if image.wcs.isCelestial():
            header = galsim.FitsHeader()
            image.wcs.writeToFitsHeader(header, image.bounds)
            border = models.parameters.nborder
            header['NAXIS1'] = image.array.shape[1] + 2 * border
            header['NAXIS2'] = image.array.shape[0] + 2 * border
            # The GWCS in the simulation extras uses zero-based raw detector
            # coordinates, including the reference-pixel border.
            header['CRPIX1'] += border - image.xmin
            header['CRPIX2'] += border - image.ymin
            # Preserve the WCS as simulation information; an L1 raw model is
            # not a calibrated image with an assigned sky solution.
            extras['wcs'] = self.wcs_from_fits_header(header.header)
        if self.include_raw_header:
            extras['raw_header'] = dict(image.header)
        with asdf.AsdfFile({'roman': raw, 'romanisim': extras}) as af:
            af.write_to(file_name)
        logger.info('Saved CMOS L1 ASDF %s', file_name)

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
        ny, nx = header["NAXIS2"], header["NAXIS1"]
        x0, y0 = w.wcs.crpix

        cd = w.wcs.piximg_matrix

        if w.sip is None:
            a, b = np.zeros((2, 2)), np.zeros((2, 2))
            a_order = b_order = 1
        else:
            cfx, cfy = np.dot(cd, [w.sip.a.ravel(), w.sip.b.ravel()])
            a = np.reshape(cfx, w.sip.a.shape)
            b = np.reshape(cfy, w.sip.b.shape)
            a_order, b_order = w.sip.a_order, w.sip.b_order
        a[1, 0] = cd[0, 0]
        a[0, 1] = cd[0, 1]
        b[1, 0] = cd[1, 0]
        b[0, 1] = cd[1, 1]

        polx = coeffs_to_poly(a, a_order)
        poly = coeffs_to_poly(b, b_order)

        # construct GWCS:
        det2sky = (
            (Shift(-x0) & Shift(-y0))
            | Mapping((0, 1, 0, 1))
            | (polx & poly)
            | Pix2Sky_TAN()
            | RotateNative2Celestial(*w.wcs.crval, w.wcs.lonpole)
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
        # Galsim uses two headers, one in the image and one in the WCS. We need to combine them to get the
        # full information.
        wcs_header.update(image.header)

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
        tree.meta.filename = Path(fname_path).name  # ===> save() appies this already. confirm!

        # tree.meta.file_date #=> don't assign value, it's set autometically at save() call
        # .meta.exposure
        tree.meta.exposure.type = "WFI_IMAGE"
        tree.meta.exposure.exposure_time = exptime
        # meta.guide_star
        # meta.instrument
        tree.meta.instrument.name = wcs_header["INSTRUME"]
        tree.meta.instrument.detector = f"{wcs_header['INSTRUME']}{sca:02}"
        tree.meta.instrument.optical_element = "F" + fltr[1:]
        # mata.observation
        # pass being a special python-statement, dot call on attr named 'pass' ends up as Error
        # tree.meta.observation["pass'] = tree.meta.observation["pass"]
        tree.meta.observation.visit = base["input"]["obseq_data"]["visit"]
        # meta.pointing
        tree.meta.pointing.target_aperture = (
            f"{wcs_header['INSTRUME']}_CEN"  # what kind of aperture is wcs_header['RA_TARG']
        )
        tree.meta.pointing.target_ra = image.wcs.center.ra.deg  # or wcs_header['RA_TARG']?
        tree.meta.pointing.target_dec = image.wcs.center.dec.deg  # or wcs_header['DEC_TARG']
        # meta.wcs
        tree.meta.wcs = self.wcs_from_fits_header(wcs_header)
        # meta.wcsinfo
        tree.meta.wcsinfo.aperture_name = (
            f"{wcs_header['INSTRUME']}{sca:02}_FULL"  # what does full stand for? # FULL or CEN?
        )
        # meta.photometry

        tree.data = image.array.astype("float32")
        tree.dq = np.zeros_like(image.array, dtype="uint32")  # Placeholder for data quality array
        tree.err = np.zeros_like(image.array, dtype="float16")  # Placeholder for error array

        _ = tree.to_asdf(fname_path)
        logger.info(f"saved {fname_path}")


RegisterOutputType("RomanASDF", RomanASDFBuilder())
