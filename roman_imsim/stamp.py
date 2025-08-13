import galsim
import galsim.config
import galsim.roman as roman
import numpy as np
from galsim.config import RegisterStampType, StampBuilder

# import os, psutil
# process = psutil.Process()


class Roman_stamp(StampBuilder):
    """This performs the tasks necessary for building the stamp for a single object.

    It uses the regular Basic functions for most things.
    It specializes the quickSkip, buildProfile, and draw methods.
    """

    _trivial_sed = galsim.SED(
        galsim.LookupTable([100, 2600], [1, 1], interpolant="linear"),
        wave_type="nm",
        flux_type="fphotons",
    )

    def setup(self, config, base, xsize, ysize, ignore, logger):
        """
        Do the initialization and setup for building a postage stamp.

        In the base class, we check for and parse the appropriate size and position values in
        config (aka base['stamp'] or base['image'].

        Values given in base['stamp'] take precedence if these are given in both places (which
        would be confusing, so probably shouldn't do that, but there might be a use case where it
        would make sense).

        Parameters:
            config:     The configuration dict for the stamp field.
            base:       The base configuration dict.
            xsize:      The xsize of the image to build (if known).
            ysize:      The ysize of the image to build (if known).
            ignore:     A list of parameters that are allowed to be in config that we can
                        ignore here. i.e. it won't be an error if these parameters are present.
            logger:     A logger object to log progress.

        Returns:
            xsize, ysize, image_pos, world_pos
        """
        # print('stamp setup',process.memory_info().rss)

        gal = galsim.config.BuildGSObject(base, "gal", logger=logger)[0]
        if gal is None:
            raise galsim.config.SkipThisObject("gal is None (invalid parameters)")
        base["object_type"] = getattr(gal, "object_type", "")
        bandpass = base["bandpass"]
        if not hasattr(gal, "flux"):
            # In this case, the object flux has not been precomputed
            # or cached by the skyCatalogs code.
            gal.flux = gal.calculateFlux(bandpass)
        self.flux = gal.flux
        # Cap (star) flux at 30M photons to avoid gross artifacts when trying
        # to draw the Roman PSF in finite time and memory
        flux_cap = 3e7
        if self.flux > flux_cap:
            if (
                hasattr(gal, "original")
                and hasattr(gal.original, "original")
                and isinstance(gal.original.original, galsim.DeltaFunction)
            ) or (isinstance(gal, galsim.DeltaFunction)):
                gal = gal.withFlux(flux_cap, bandpass)
                self.flux = flux_cap
                gal.flux = flux_cap
        base["flux"] = gal.flux
        base["mag"] = -2.5 * np.log10(gal.flux) + bandpass.zeropoint
        # print('stamp setup2',process.memory_info().rss)

        # Compute or retrieve the realized flux.
        self.rng = galsim.config.GetRNG(config, base, logger, "Roman_stamp")
        self.realized_flux = galsim.PoissonDeviate(self.rng, mean=self.flux)()
        base["realized_flux"] = self.realized_flux

        # Check if the realized flux is 0.
        if self.realized_flux == 0:
            # If so, we'll skip everything after this.
            # The mechanism within GalSim to do this is to raise a special SkipThisObject class.
            raise galsim.config.SkipThisObject("realized flux=0")

        # Otherwise figure out the stamp size
        if self.flux < 10:
            # For really faint things, don't try too hard.  Just use 32x32.
            image_size = 32
            self.pupil_bin = "achromatic"

        else:
            gal_achrom = gal.evaluateAtWavelength(bandpass.effective_wavelength)
            if hasattr(gal_achrom, "original") and isinstance(gal_achrom.original, galsim.DeltaFunction):
                # For bright stars, set the following stamp size limits
                if self.flux < 1e6:
                    image_size = 500
                    self.pupil_bin = 8
                elif self.flux < 6e6:
                    image_size = 1000
                    self.pupil_bin = 4
                else:
                    image_size = 1600
                    self.pupil_bin = 2
            else:
                self.pupil_bin = 8
                # # Get storead achromatic PSF
                # psf = galsim.config.BuildGSObject(base, 'psf', logger=logger)[0]['achromatic']
                # obj = galsim.Convolve(gal_achrom, psf).withFlux(self.flux)
                obj = gal_achrom.withGSParams(galsim.GSParams(stepk_minimum_hlr=20))
                image_size = obj.getGoodImageSize(roman.pixel_scale)

        # print('stamp setup3',process.memory_info().rss)
        base["pupil_bin"] = self.pupil_bin
        logger.info("Object flux is %d", self.flux)
        logger.info("Object %d will use stamp size = %s", base.get("obj_num", 0), image_size)

        # Determine where this object is going to go:
        # This is the same as what the base StampBuilder does:
        if "image_pos" in config:
            image_pos = galsim.config.ParseValue(config, "image_pos", base, galsim.PositionD)[0]
        else:
            image_pos = None

        if "world_pos" in config:
            world_pos = galsim.config.ParseWorldPos(config, "world_pos", base, logger)
        else:
            world_pos = None

        return image_size, image_size, image_pos, world_pos

    def buildPSF(self, config, base, gsparams, logger):
        """Build the PSF object.

        For the Basic stamp type, this builds a PSF from the base['psf'] dict, if present,
        else returns None.

        Parameters:
            config:     The configuration dict for the stamp field.
            base:       The base configuration dict.
            gsparams:   A dict of kwargs to use for a GSParams.  More may be added to this
                        list by the galaxy object.
            logger:     A logger object to log progress.

        Returns:
            the PSF
        """
        if base.get("psf", {}).get("type", "roman_psf") != "roman_psf":
            return galsim.config.BuildGSObject(base, "psf", gsparams=gsparams, logger=logger)[0]

        roman_psf = galsim.config.GetInputObj("roman_psf", config, base, "buildPSF")
        psf = roman_psf.getPSF(self.pupil_bin, base["image_pos"])
        return psf

    def getDrawMethod(self, config, base, logger):
        """Determine the draw method to use.

        @param config       The configuration dict for the stamp field.
        @param base         The base configuration dict.
        @param logger       A logger object to log progress.

        @returns method
        """
        method = galsim.config.ParseValue(config, "draw_method", base, str)[0]
        # Here we add the the custom "fft_poisson" method to the list of valid methods.
        valid_methods = galsim.config.valid_draw_methods + ('fft_poisson',)
        if method not in valid_methods:
            raise galsim.GalSimConfigValueError(
                "Invalid draw_method.", method, valid_methods
            )
        if method == "auto":
            if self.pupil_bin in [4, 2]:
                logger.info("Auto -> Use FFT drawing for object %d.", base["obj_num"])
                return "fft_poisson"
            else:
                logger.info("Auto -> Use photon shooting for object %d.", base["obj_num"])
                return "phot"
        else:
            # If user sets something specific for the method, rather than auto,
            # then respect their wishes.
            logger.info("Use specified method=%s for object %d.", method, base["obj_num"])
            return method

    @classmethod
    def _fix_seds_24(cls, prof, bandpass):
        # If any SEDs are not currently using a LookupTable for the function or if they are
        # using spline interpolation, then the codepath is quite slow.
        # Better to fix them before doing WavelengthSampler.
        if isinstance(prof, galsim.ChromaticObject):
            wave_list, _, _ = galsim.utilities.combine_wave_list(prof.SED, bandpass)
            sed = prof.SED
            # TODO: This bit should probably be ported back to Galsim.
            #       Something like sed.make_tabulated()
            if not isinstance(sed._spec, galsim.LookupTable) or sed._spec.interpolant != "linear":
                # Workaround for https://github.com/GalSim-developers/GalSim/issues/1228
                f = np.broadcast_to(sed(wave_list), wave_list.shape)
                new_spec = galsim.LookupTable(wave_list, f, interpolant="linear")
                new_sed = galsim.SED(new_spec, "nm", "fphotons" if sed.spectral else "1")
                prof.SED = new_sed

            # Also recurse onto any components.
            if hasattr(prof, "obj_list"):
                for obj in prof.obj_list:
                    cls._fix_seds_24(obj, bandpass)
            if hasattr(prof, "original"):
                cls._fix_seds_24(prof.original, bandpass)

    @classmethod
    def _fix_seds_25(cls, prof, bandpass):
        # If any SEDs are not currently using a LookupTable for the function or if they are
        # using spline interpolation, then the codepath is quite slow.
        # Better to fix them before doing WavelengthSampler.

        # In GalSim 2.5, SEDs are not necessarily constructed in most chromatic objects.
        # And really the only ones we need to worry about are the ones that come from
        # SkyCatalog, since they might not have linear interpolants.
        # Those objects are always SimpleChromaticTransformations.  So only fix those.
        if isinstance(prof, galsim.SimpleChromaticTransformation) and (
            not isinstance(prof._flux_ratio._spec, galsim.LookupTable)
            or prof._flux_ratio._spec.interpolant != "linear"
        ):
            sed = prof._flux_ratio
            wave_list, _, _ = galsim.utilities.combine_wave_list(sed, bandpass)
            f = np.broadcast_to(sed(wave_list), wave_list.shape)
            new_spec = galsim.LookupTable(wave_list, f, interpolant="linear")
            new_sed = galsim.SED(new_spec, "nm", "fphotons" if sed.spectral else "1")
            prof._flux_ratio = new_sed

        # Also recurse onto any components.
        if isinstance(prof, galsim.ChromaticObject):
            if hasattr(prof, "obj_list"):
                for obj in prof.obj_list:
                    cls._fix_seds_25(obj, bandpass)
            if hasattr(prof, "original"):
                cls._fix_seds_25(prof.original, bandpass)

    def draw(self, prof, image, method, offset, config, base, logger):
        """Draw the profile on the postage stamp image.

        Parameters:
            prof:       The profile to draw.
            image:      The image onto which to draw the profile (which may be None).
            method:     The method to use in drawImage.
            offset:     The offset to apply when drawing.
            config:     The configuration dict for the stamp field.
            base:       The base configuration dict.
            logger:     A logger object to log progress.

        Returns:
            the resulting image
        """
        if prof is None:
            # If was decide to do any rejection steps, this could be set to None, in which case,
            # don't draw anything.
            return image

        # Prof is normally a convolution here with obj_list being [gal, psf1, psf2,...]
        # for some number of component PSFs.
        # print('stamp draw',process.memory_info().rss)

        gal, *psfs = prof.obj_list if hasattr(prof, "obj_list") else [prof]

        faint = self.flux < 40
        bandpass = base["bandpass"]
        if faint:
            logger.info("Flux = %.0f  Using trivial sed", self.flux)
        else:
            self.fix_seds(gal, bandpass)

        image.wcs = base["wcs"]

        # Set limit on the size of photons batches to consider when
        # calling gsobject.drawImage.
        maxN = int(1e6)
        if "maxN" in config:
            maxN = galsim.config.ParseValue(config, "maxN", base, int)[0]
        # print('stamp draw2',process.memory_info().rss)

        if method == "fft" or method == "fft_poisson":
            fft_image = image.copy()
            fft_offset = offset
            kwargs = dict(
                method="fft",
                offset=fft_offset,
                image=fft_image,
            )
            if not faint and config.get("fft_photon_ops"):
                kwargs.update(
                    {
                        "photon_ops": galsim.config.BuildPhotonOps(config, "fft_photon_ops", base, logger),
                        "maxN": maxN,
                        "rng": self.rng,
                        "n_subsample": 1,
                    }
                )

            # Go back to a combined convolution for fft drawing.
            gal = gal.withFlux(self.flux, bandpass)
            prof = galsim.Convolve([gal] + psfs)
            try:
                prof.drawImage(bandpass, **kwargs)
            except galsim.errors.GalSimFFTSizeError as e:
                # I think this shouldn't happen with the updates I made to how the image size
                # is calculated, even for extremely bright things.  So it should be ok to
                # just report what happened, give some extra information to diagonose the problem
                # and raise the error.
                logger.error("Caught error trying to draw using FFT:")
                logger.error("%s", e)
                logger.error("You may need to add a gsparams field with maximum_fft_size to")
                logger.error("either the psf or gal field to allow larger FFTs.")
                logger.info("prof = %r", prof)
                logger.info("fft_image = %s", fft_image)
                logger.info("offset = %r", offset)
                logger.info("wcs = %r", image.wcs)
                raise
            if method == "fft_poisson":
                # Some pixels can end up negative from FFT numerics.  Just set them to 0.
                fft_image.array[fft_image.array < 0] = 0.0
                fft_image.addNoise(galsim.PoissonNoise(rng=self.rng))
            # In case we had to make a bigger image, just copy the part we need.
            image += fft_image[image.bounds]

        else:
            # We already calculated realized_flux above.  Use that now and tell GalSim not
            # recalculate the Poisson realization of the flux.
            gal = gal.withFlux(self.realized_flux, bandpass)
            # print('stamp draw3b ',process.memory_info().rss)

            if not faint and "photon_ops" in config:
                photon_ops = galsim.config.BuildPhotonOps(config, "photon_ops", base, logger)
            else:
                photon_ops = []

            # Put the psfs at the start of the photon_ops.
            # Probably a little better to put them a bit later than the start in some cases
            # (e.g. after TimeSampler, PupilAnnulusSampler), but leave that as a todo for now.
            photon_ops = psfs + photon_ops

            # prof = galsim.Convolve([gal] + psfs)

            # print('-------- gal ----------',gal)
            # print('-------- psf ----------',psfs)

            # print('stamp draw3a',process.memory_info().rss)
            gal.drawImage(
                bandpass,
                method="phot",
                offset=offset,
                rng=self.rng,
                maxN=maxN,
                n_photons=self.realized_flux,
                image=image,
                photon_ops=photon_ops,
                sensor=None,
                add_to_image=True,
                poisson_flux=False,
            )
        # print('stamp draw3',process.memory_info().rss)

        return image


# Pick the right function to be _fix_seds.
if galsim.__version_info__ < (2, 5):
    Roman_stamp.fix_seds = Roman_stamp._fix_seds_24
else:
    Roman_stamp.fix_seds = Roman_stamp._fix_seds_25


# Register this as a valid type
RegisterStampType("Roman_stamp", Roman_stamp())
