from importlib.metadata import version, PackageNotFoundError
import galsim
import galsim.config
import romanisim.models as models
import numpy as np
from astropy.time import Time
from galsim.config import RegisterImageType
from galsim.config.image_scattered import ScatteredImageBuilder
from galsim.image import Image

from .ramp import CMOSRamp, CMOSReadout, validate_read_pattern


class RomanSCAImageBuilderBase(ScatteredImageBuilder):
    """Shared setup, detector image creation, positions, and batching for Roman SCAs."""

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
            "use_fft_bright": bool,
        }
        params = galsim.config.GetAllParams(config, base, req=req, opt=opt, ignore=ignore + extra_ignore)[0]

        self.sca = params["SCA"]
        base["SCA"] = self.sca
        self.filter = params["filter"]
        self.mjd = params["mjd"]
        self.exptime = params["exptime"]

        # If draw_method isn't in image field, it may be in stamp.  Check.
        self.draw_method = params.get("draw_method", base.get("stamp", {}).get("draw_method", "phot"))

        # If user hasn't overridden the bandpass to use, get the standard one.
        if "bandpass" not in config:
            base["bandpass"] = galsim.config.BuildBandpass(base["image"], "bandpass", base, logger=logger)

        return models.parameters.n_pix, models.parameters.n_pix

    def _makeImage(self, base):
        """Create the detector image and its common Roman metadata."""
        full_xsize = base["image_xsize"]
        full_ysize = base["image_ysize"]
        wcs = base["wcs"]

        full_image = Image(full_xsize, full_ysize, dtype=float)
        full_image.setOrigin(base["image_origin"])
        full_image.wcs = wcs

        full_image.setZero()

        full_image.header = galsim.FitsHeader()
        try:
            full_image.header["VERSION"] = version("roman_imsim")
        except PackageNotFoundError:
            full_image.header["VERSION"] = "unknown"
        # Some of these should be in the WCS creation but it will be changed in the future so we leave it
        # here for now.
        full_image.header["NAXIS"] = 2
        full_image.header["NAXIS1"] = full_xsize
        full_image.header["NAXIS2"] = full_ysize
        full_image.header["RADESYS"] = "ICRS"
        full_image.header["SCA"] = self.sca
        # This should go to wcs header
        full_image.header["INSTRUME"] = "WFI"
        full_image.header["EXPTIME"] = self.exptime
        full_image.header["MJD-OBS"] = self.mjd
        full_image.header["DATE-OBS"] = Time(self.mjd, format="mjd").datetime.isoformat()
        full_image.header["FILTER"] = self.filter
        full_image.header["ZPTMAG"] = 2.5 * np.log10(self.exptime * models.parameters.collecting_area)

        base["current_image"] = full_image
        return full_image

    def _setupPositions(self, config, base):
        """Validate object positions or supply uniform detector positions."""
        full_xsize = base["image_xsize"]
        full_ysize = base["image_ysize"]
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

    def _iterObjectBatches(self):
        """Yield existing batch indices, counts, and half-open object ranges."""
        nbatch = self.nobjects // 1000 + 1
        for batch in range(nbatch):
            start_obj_num = self.nobjects * batch // nbatch
            end_obj_num = self.nobjects * (batch + 1) // nbatch
            yield batch, nbatch, start_obj_num, end_obj_num


class RomanSCAImageBuilder(RomanSCAImageBuilderBase):
    """Accumulate full-exposure image stamps using shared Roman SCA setup."""

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
        full_image = self._makeImage(base)
        self._setupPositions(config, base)
        for batch, nbatch, start_obj_num, end_obj_num in self._iterObjectBatches():
            # Calculate the number of objects in this batch
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
                # imprint the stemp of each object in this loop
                full_image[bounds] += stamps[k][bounds]
            stamps = None

        # # Bring the image so far up to a flat noise variance
        # current_var = FlattenNoiseVariance(
        #         base, full_image, stamps, current_vars, logger)

        return full_image, None


class RomanSCAImageBuilderCMOS(RomanSCAImageBuilderBase):
    """Sample all sources for each frame and retain nondestructive L1 resultants."""

    def setup(self, config, base, image_num, obj_num, ignore, logger):
        if 'nproc' in config and galsim.config.ParseValue(config, 'nproc', base, int)[0] != 1:
            raise galsim.GalSimConfigError(
                'CMOS frame sampling requires image.nproc: 1 to retain object RNG streams')
        return super().setup(config, base, image_num, obj_num,
                             ignore + ['pedestal', 'pedestal_extra_noise', 'saturation', 'reference_read'], logger)

    def buildImage(self, config, base, image_num, obj_num, logger):
        rdata = galsim.config.GetInputObj('resultant_data', config, base, 'CMOS ramp')
        pattern = validate_read_pattern(rdata.get('strategy'))
        if not np.isfinite(self.exptime) or self.exptime <= 0:
            raise galsim.GalSimConfigError('CMOS exposure time must be finite and positive')
        if base['stamp']['type'] != 'Roman_stamp_CMOS':
            raise galsim.GalSimConfigError('roman_sca_cmos requires stamp.type: Roman_stamp_CMOS')
        frame_time = self.exptime / pattern[-1][-1]
        charge = self._makeImage(base)
        self._setupPositions(config, base)
        # Exposure-local state: preserve each object's RNG across intervals but
        # never reuse charge, samplers, or a previous exposure's resultants.
        base.pop('_cmos_ramp', None)
        base['_cmos_object_rngs'] = {}
        base['_cmos_dt'] = frame_time
        rng = galsim.config.GetRNG(config, base, logger, 'CMOS readout')
        readout = CMOSReadout(config, base, charge, rng, logger)
        want_reference = galsim.config.ParseValue(config, 'reference_read', base, bool)[0] if 'reference_read' in config else False
        reference = readout.read(charge) if want_reference else None
        group_for_read = {read: i for i, group in enumerate(pattern) for read in group}
        resultants = np.zeros((len(pattern),) + charge.array.shape, dtype='f8')
        for frame in range(1, pattern[-1][-1] + 1):
            for batch, nbatch, start, end in self._iterObjectBatches():
                stamps, _ = galsim.config.BuildStamps(
                    end - start, base, logger=logger, obj_num=obj_num + start, do_noise=False)
                base['index_key'] = 'image_num'
                for stamp in stamps:
                    if stamp is None:
                        continue
                    bounds = stamp.bounds & charge.bounds
                    if bounds.isDefined():
                        charge[bounds] += stamp[bounds]
            readout.accumulate_background(charge, frame_time)
            charge.quantize()  # remove floating roundoff from unit-electron photon weights
            if frame in group_for_read:
                i = group_for_read[frame]
                resultants[i] += readout.read(charge) / len(pattern[i])
        if reference is not None:
            all_reads = np.concatenate([reference[None], resultants])
            cube, all_dq = readout.digitize(all_reads, [[0]] + pattern, frame_time)
            reference, cube, dq = cube[0], cube[1:], all_dq[1:]
        else:
            cube, dq = readout.digitize(resultants, pattern, frame_time)
        base['_cmos_ramp'] = CMOSRamp(cube, dq, pattern, frame_time, readout.gain, reference)
        base.pop('_cmos_dt', None)
        base.pop('_cmos_object_rngs', None)
        return charge, None

    def addNoise(self, image, config, base, image_num, obj_num, current_var, logger):
        # Read electronics have already been applied to resultant copies.
        pass


# Register this as a valid type
RegisterImageType("roman_sca_cmos", RomanSCAImageBuilderCMOS())
# Register this as a valid type
RegisterImageType("roman_sca", RomanSCAImageBuilder())
