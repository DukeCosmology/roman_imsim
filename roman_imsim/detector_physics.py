import os

import fitsio as fio
import galsim as galsim
import galsim.roman as roman
import numpy as np
from galsim.config import ParseValue, ReadYaml

from roman_imsim.obseq import ObSeqDataLoader

sca_number_to_file = {
    1: "SCA_22066_211227_v001.fits",
    2: "SCA_21815_211221_v001.fits",
    3: "SCA_21946_211225_v001.fits",
    4: "SCA_22073_211229_v001.fits",
    5: "SCA_21816_211222_v001.fits",
    6: "SCA_20663_211102_v001.fits",
    7: "SCA_22069_211228_v001.fits",
    8: "SCA_21641_211216_v001.fits",
    9: "SCA_21813_211219_v001.fits",
    10: "SCA_22078_211230_v001.fits",
    11: "SCA_21947_211226_v001.fits",
    12: "SCA_22077_211230_v001.fits",
    13: "SCA_22067_211227_v001.fits",
    14: "SCA_21814_211220_v001.fits",
    15: "SCA_21645_211228_v001.fits",
    16: "SCA_21643_211218_v001.fits",
    17: "SCA_21319_211211_v001.fits",
    18: "SCA_20833_211116_v001.fits",
}


def write_fits(old_filename, new_filename, img, err, dq, sca, sky_mean=None):
    from astropy.io import fits

    hdul = fits.open(old_filename)  # open a FITS file
    hdr = hdul[0].header

    if sky_mean is not None:
        hdr["SKY_MEAN"] = sky_mean

    fit0 = fits.PrimaryHDU(header=hdr)
    fit1 = fits.ImageHDU(data=img.array, header=hdr, name="SCI", ver=1)
    if err is not None:
        fit2 = fits.ImageHDU(data=err.array, header=hdr, name="ERR", ver=1)
    if dq is not None:
        fit3 = fits.ImageHDU(data=dq, header=hdr, name="DQ", ver=1)
    if dq is not None:
        if err is not None:
            new_fits_file = fits.HDUList([fit0, fit1, fit2, fit3])
        else:
            new_fits_file = fits.HDUList([fit0, fit1, fit3])
    elif err is not None:
        new_fits_file = fits.HDUList([fit0, fit1, fit2])
    else:
        new_fits_file = fits.HDUList([fit0, fit1])
    new_fits_file.writeto(new_filename, overwrite=True)

    return


class get_pointing(object):
    """
    Class to store stuff about the telescope
    """

    def __init__(self, params, visit, SCA):

        file_name = params["input"]["obseq_data"]["file_name"]
        obseq_data = ObSeqDataLoader(file_name, visit, SCA, logger=None)
        self.filter = obseq_data.ob["filter"]
        self.sca = obseq_data.ob["sca"]
        self.visit = obseq_data.ob["visit"]
        self.date = obseq_data.ob["date"]
        self.exptime = obseq_data.ob["exptime"]
        self.bpass = roman.getBandpasses()[self.filter]
        self.WCS = roman.getWCS(
            world_pos=galsim.CelestialCoord(ra=obseq_data.ob["ra"], dec=obseq_data.ob["dec"]),
            PA=obseq_data.ob["pa"],
            date=self.date,
            SCAs=self.sca,
            PA_is_FPA=True,
        )[self.sca]
        self.radec = self.WCS.toWorld(galsim.PositionI(int(roman.n_pix / 2), int(roman.n_pix / 2)))


class modify_image(object):
    """
    Class to simulate non-idealities and noise of roman detector images.
    """

    def __init__(self, params, visit, sca, dither_from_file, sca_filepath=None, use_galsim=False):
        """
        Set up noise properties of image

        Input
        params  : parameter dict
        rng     : Random generator
        """

        self.params = ReadYaml(params)[0]
        if "max_sun_angle" in self.params["image"]["wcs"]:
            roman.max_sun_angle = self.params["image"]["wcs"]["max_sun_angle"]
            roman.roman_wcs.max_sun_angle = self.params["image"]["wcs"]["max_sun_angle"]
        self.params["save_diff"] = False
        self.params["dither_from_file"] = dither_from_file

        self.pointing = get_pointing(self.params, visit, sca)
        roman.exptime = self.pointing.exptime

        # Load sca file if applicable
        if sca_filepath is not None:
            self.df = fio.FITS(sca_filepath + "/" + sca_number_to_file[self.pointing.sca])
            print("------- Using SCA files --------")
        else:
            self.df = None
            print("------- Using simple detector model --------")

        self.params["output"]["file_name"]["items"] = [self.pointing.filter, visit, sca]
        imfilename = ParseValue(self.params["output"], "file_name", self.params, str)[0]

        old_filename = os.path.join(self.params["output"]["dir"], imfilename)
        if not os.path.exists(
            self.params["output"]["dir"].replace("truth", self.get_path_name(use_galsim=use_galsim))
        ):
            os.mkdir(self.params["output"]["dir"].replace("truth", self.get_path_name(use_galsim=use_galsim)))
        new_filename = os.path.join(self.params["output"]["dir"], imfilename).replace(
            "truth", self.get_path_name(use_galsim=use_galsim)
        )

        b = galsim.BoundsI(xmin=1, ymin=1, xmax=roman.n_pix, ymax=roman.n_pix)
        im = fio.FITS(old_filename)[-1].read()
        im = galsim.Image(im, bounds=b, wcs=self.pointing.WCS)

        rng = galsim.BaseDeviate(visit * sca)
        force_cvz = False
        if "force_cvz" in self.params["image"]["wcs"]:
            if self.params["image"]["wcs"]["force_cvz"]:
                force_cvz = True
        self.setup_sky(im, self.pointing, rng, visit * sca, force_cvz=force_cvz)

        img, err, dq, sky_mean, sky_noise = self.add_effects(im, None, self.pointing, use_galsim=use_galsim)

        write_fits(
            old_filename,
            new_filename,
            img,
            sky_noise,
            dq,
            self.pointing.sca,
            sky_mean=sky_mean,
        )

    def get_path_name(self, use_galsim=False):

        if self.df is not None:
            return "sca_model"
        elif use_galsim:
            return "galsim_model"
        else:
            return "simple_model"

    def add_effects(self, im, wt, pointing, use_galsim=False):

        if self.df is not None:
            return self.add_effects_scafile(im, wt, pointing)
        elif use_galsim:
            return self.add_effects_galsim(im, wt, pointing)
        else:
            return self.add_effects_simple(im, wt, pointing)

    def add_effects_scafile(self, im, wt, pointing):
        """
        Add detector effects for Roman.
        Input:
        im        : Postage stamp or image.
        pointing  : Pointing object
        radec     : World coordinate position of image
        local_wcs : The local WCS
        phot      : photon shooting mode
        Preserve order:
        1) qe
        2) brighter-fatter
        3) persistence
        4) quantize
        5) dark current
        6) saturation
        7) CNL
        8) IPC
        9) dead pixel mask
        10) vertical trailing pixel effect
        11) read noise (e-)
        12) gain (in unit of e/adu)
        13) bias
        14) quantize
        Take 4088x4088 sky image as input
        Pad the image to 4096x4096
        Output 4088x4088 images in uint16
        """

        # check input dimension
        if not im.array.shape == (4088, 4088):
            raise ValueError("input image for detector effects must be 4088x4088.")

        im = self.add_background(im)  # Add background to image and save background

        # create padded image
        bound_pad = galsim.BoundsI(xmin=1, ymin=1, xmax=4096, ymax=4096)
        im_pad = galsim.Image(bound_pad)
        im_pad.array[4:-4, 4:-4] = im.array[:, :]

        self.set_diff(im_pad)

        im_pad = self.qe(im_pad)
        self.diff("qe", im_pad)

        im_pad = self.bfe(im_pad)
        self.diff("bfe", im_pad)

        im_pad = self.add_persistence(im_pad, pointing)
        self.diff("pers", im_pad)

        im_pad.quantize()
        self.diff("quantize1", im_pad)

        im_pad = self.dark_current(im_pad)
        self.diff("dark", im_pad)

        im_pad = self.saturate(im_pad)
        self.diff("sat", im_pad)

        im_pad = self.nonlinearity(im_pad)
        self.diff("cnl", im_pad)

        im_pad = self.interpix_cap(im_pad)
        self.diff("ipc", im_pad)

        im_pad = self.deadpix(im_pad)
        self.diff("deadpix", im_pad)

        im_pad = self.vtpe(im_pad)
        self.diff("vtpe", im_pad)

        im_pad = self.add_read_noise(im_pad)
        self.diff("read", im_pad)

        im_pad = self.add_gain(im_pad)
        self.diff("gain", im_pad)

        im_pad = self.add_bias(im_pad)
        self.diff("bias", im_pad)

        im_pad.quantize()
        self.diff("quantize2", im_pad)

        # output 4088x4088 img in uint16
        im.array[:, :] = im_pad.array[4:-4, 4:-4]
        im = galsim.Image(im, dtype=np.uint16)

        # data quality image
        # 0x1 -> non-responsive
        # 0x2 -> hot pixel
        # 0x4 -> very hot pixel
        # 0x8 -> adjacent to pixel with strange response
        # 0x10 -> low CDS, high total noise pixel (may have strange settling
        #         behaviors, not recommended for precision applications)
        # 0x20 -> CNL fit went down to the minimum number of points
        #         (remaining degrees of freedom = 0)
        # 0x40 -> no solid-waffle solution for this region (set gain value to
        #         array median). normally occurs in a few small regions of some
        #         SCAs with lots of bad pixels. [recommend not to use these
        #         regions for WL analysis]
        # 0x80 -> wt==0
        dq = self.df["BADPIX"][4:4092, 4:4092]
        # get weight map
        if wt is not None:
            dq[wt == 0] += 128

        sky_noise = self.sky.copy()
        sky_noise = self.finalize_sky_im(sky_noise, pointing)

        return (
            im,
            self.sky[self.sky.bounds & im.bounds] - self.sky_mean,
            dq,
            self.sky_mean,
            sky_noise,
        )
        return im, self.sky[self.sky.bounds & im.bounds] - self.sky_mean, dq, self.sky_mean, sky_noise

    def add_effects_galsim(self, im, wt, pointing):
        """
        Add detector effects for Roman.

        Input:
        im        : Postage stamp or image.
        pointing  : Pointing object
        radec     : World coordinate position of image
        local_wcs : The local WCS
        phot      : photon shooting mode

        Preserve order:
        1) add_background
        2) add_poisson_noise
        3) recip_failure
        4) quantize
        5) dark_current
        6) add_persistence
        7) nonlinearity
        8) interpix_cap
        9) Read noise
        10) e_to_ADU
        11) quantize

        """

        # check input dimension
        if not im.array.shape == (4088, 4088):
            raise ValueError("input image for detector effects must be 4088x4088.")

        im = self.add_background(im)  # Add background to image and save background
        # im = self.add_poisson_noise(im,sky_image,phot=phot) # Add poisson noise to image
        im = self.recip_failure(im)  # Introduce reciprocity failure to image
        im.quantize()
        # At this point in the image generation process, an integer number of photons gets detected
        im = self.dark_current(im)  # Add dark current to image
        im = self.add_persistence(im, pointing)
        im = self.saturate(im)
        im = self.nonlinearity(im)  # Apply nonlinearity
        im = self.interpix_cap(im)  # Introduce interpixel capacitance to image.
        im = self.add_read_noise(im)
        im = self.e_to_ADU(im)  # Convert electrons to ADU
        im.quantize()  # Finally, the analog-to-digital converter reads in an integer value.
        # Note that the image type after this step is still a float. If we want to actually
        # get integer values, we can do new_img = galsim.Image(im, dtype=int)
        # Since many people are used to viewing background-subtracted images, we return a
        # version with the background subtracted (also rounding that to an int).
        sky_noise = self.sky.copy()
        sky_noise = self.finalize_sky_im(sky_noise, pointing)
        # im = galsim.Image(im, dtype=int)
        # get weight map
        # sky_image.invertSelf()

        dq = np.zeros(im.array.shape, dtype=np.uint32)

        if wt is not None:
            dq[wt == 0] += 2

        return (
            im,
            self.sky[self.sky.bounds & im.bounds] - self.sky_mean,
            dq,
            self.sky_mean,
            sky_noise,
        )

    def add_effects_simple(self, im, wt, pointing):
        """
        Add detector effects for Roman.

        Input:
        im        : Postage stamp or image.
        pointing  : Pointing object
        radec     : World coordinate position of image
        local_wcs : The local WCS
        phot      : photon shooting mode

        Preserve order:
        1) add_background
        2) add_poisson_noise
        3) recip_failure
        4) quantize
        5) dark_current
        6) add_persistence
        7) nonlinearity
        8) interpix_cap
        9) Read noise
        10) e_to_ADU
        11) quantize
        """

        # check input dimension
        if not im.array.shape == (4088, 4088):
            raise ValueError("input image for detector effects must be 4088x4088.")

        im = self.add_background(im)  # Add background to image and save background
        # im = self.add_poisson_noise(im,sky_image,phot=phot) # Add poisson noise to image
        im.quantize()
        # At this point in the image generation process, an integer number of photons gets detected
        im = self.dark_current(im)  # Add dark current to image
        im = self.saturate(im)
        im = self.add_read_noise(im)
        im = self.e_to_ADU(im)  # Convert electrons to ADU
        im.quantize()  # Finally, the analog-to-digital converter reads in an integer value.
        # Note that the image type after this step is still a float. If we want to actually
        # get integer values, we can do new_img = galsim.Image(im, dtype=int)
        # Since many people are used to viewing background-subtracted images, we return a
        # version with the background subtracted (also rounding that to an int).
        # im,sky_image = self.finalize_background_subtract(im,sky_image)
        # im = galsim.Image(im, dtype=int)
        # get weight map
        # sky_image.invertSelf()

        sky_noise = self.sky.copy()
        sky_noise = self.finalize_sky_im(sky_noise, pointing)

        # nan check
        dq = np.zeros(im.array.shape, dtype=np.uint32)
        if wt is not None:
            dq[wt == 0] += 2

        return (
            im,
            self.sky[self.sky.bounds & im.bounds] - self.sky_mean,
            dq,
            self.sky_mean,
            sky_noise,
        )
        return im, self.sky[self.sky.bounds & im.bounds] - self.sky_mean, dq, self.sky_mean, sky_noise

    def set_diff(self, im=None):
        if self.params["save_diff"]:
            self.pre = im.copy()
            self.pre.write("bg.fits", dir=self.params["diff_dir"])
        return

    def diff(self, msg, im=None, verbose=True):
        if self.params["save_diff"]:
            diff = im - self.pre
            diff.write("%s_diff.fits" % msg, dir=self.params["diff_dir"])
            self.pre = im.copy()
            im.write("%s_cumul.fits" % msg, dir=self.params["diff_dir"])
        return

    def qe(self, im):
        """
        Apply the wavelength-independent relative QE to the image.
        Input
        im                  : Image
        RELQE1[4096,4096]   : relative QE map
        """

        im.array[:, :] *= self.df["RELQE1"][:, :]  # 4096x4096 array
        return im

    def bfe(self, im):
        """
        Apply brighter-fatter effect.
        Brighter fatter effect is a non-linear effect that deflects photons due to the
        the eletric field built by the accumulated charges. This effect exists in both
        CCD and CMOS detectors and typically percent level change in charge.
        The built-in electric field by the charges in pixels tends to repulse charges
        to nearby pixels. Thus, the profile of more illuminous ojbect becomes broader.
        This effect can also be understood effectly as change in pixel area and pixel
        boundaries.
        BFE is defined in terms of the Antilogus coefficient kernel of total pixel area change
        in the detector effect charaterization file. Kernel of the total pixel area, however,
        is not sufficient. Image simulation of the brighter fatter effect requires the shift
        of the four pixel boundaries. Before we get better data, we solve for the boundary
        shift components from the kernel of total pixel area by assumming several symmetric constraints.
        Input
        im                                      : Image
        BFE[nbfe+Delta y, nbfe+Delta x, y, x]   : bfe coefficient kernel, nbfe=2
        """

        nbfe = 2  # kernel of bfe in shape (2 x nbfe+1)*(2 x nbfe+1)
        bin_size = 128
        n_max = 32
        m_max = 32
        num_grids = 4
        n_sub = n_max // num_grids
        m_sub = m_max // num_grids

        # =======================================================================
        #      solve boundary shfit kernel aX components
        # =======================================================================
        a_area = self.df["BFE"][:, :, :, :]  # 5x5x32x32
        a_components = np.zeros((4, 2 * nbfe + 1, 2 * nbfe + 1, n_max, m_max))  # 4x5x5x32x32

        # solve aR aT aL aB for each a
        for n in range(n_max):  # m_max and n_max = 32 (binned in 128x128)
            for m in range(m_max):
                a = a_area[:, :, n, m]  # a in (2 x nbfe+1)*(2 x nbfe+1)

                #  assume two parity symmetries
                a = (a + np.fliplr(a) + np.flipud(a) + np.flip(a)) / 4.0

                r = 0.5 * (3.25 / 4.25) ** (1.5) / 1.5  # source-boundary projection
                B = (a[2, 2], a[3, 2], a[2, 3], a[3, 3], a[4, 2], a[2, 4], a[3, 4], a[4, 4])

                A = np.array(
                    [
                        [-2, -2, 0, 0, 0, 0, 0],
                        [0, 1, 0, -1, -2, 0, 0],
                        [1, 0, -1, 0, -2, 0, 0],
                        [0, 0, 0, 0, 2, -2, 0],
                        [0, 0, 0, 1, 0, -2 * r, 0],
                        [0, 0, 1, 0, 0, -2 * r, 0],
                        [0, 0, 0, 0, 0, 1 + r, -1],
                        [0, 0, 0, 0, 0, 0, 2],
                    ]
                )

                s1, s2, s3, s4, s5, s6, s7 = np.linalg.lstsq(A, B, rcond=None)[0]

                aR = np.array(
                    [
                        [0.0, -s7, -r * s6, r * s6, s7],
                        [0.0, -s6, -s5, s5, s6],
                        [0.0, -s3, -s1, s1, s3],
                        [0.0, -s6, -s5, s5, s6],
                        [0.0, -s7, -r * s6, r * s6, s7],
                    ]
                )

                aT = np.array(
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [-s7, -s6, -s4, -s6, -s7],
                        [-r * s6, -s5, -s2, -s5, -r * s6],
                        [r * s6, s5, s2, s5, r * s6],
                        [s7, s6, s4, s6, s7],
                    ]
                )

                aL = aR[::-1, ::-1]
                aB = aT[::-1, ::-1]

                a_components[0, :, :, n, m] = aR[:, :]
                a_components[1, :, :, n, m] = aT[:, :]
                a_components[2, :, :, n, m] = aL[:, :]
                a_components[3, :, :, n, m] = aB[:, :]

        # =============================
        #  Apply bfe to image
        # =============================

        #  pad and expand kernels
        # The img is clipped by the saturation level here to cap the brighter fatter effect
        # and avoid unphysical behavior

        array_pad = self.saturate(im.copy()).array[4:-4, 4:-4]  # img of interest 4088x4088
        array_pad = np.pad(
            array_pad, [(4 + nbfe, 4 + nbfe), (4 + nbfe, 4 + nbfe)], mode="symmetric"
        )  # 4100x4100 array

        dQ_components = np.zeros(
            (4, bin_size * n_max, bin_size * m_max)
        )  # (4, 4096, 4096) in order of [aR, aT, aL, aB]

        # run in sub grids to reduce memory

        # pad and expand kernels
        t = np.zeros((bin_size * n_sub, n_sub))
        for row in range(t.shape[0]):
            t[row, row // (bin_size)] = 1

        for gj in range(num_grids):
            for gi in range(num_grids):

                a_components_pad = np.zeros(
                    (
                        4,
                        2 * nbfe + 1,
                        2 * nbfe + 1,
                        bin_size * n_sub + 2 * nbfe,
                        bin_size * m_sub + 2 * nbfe,
                    )
                )  # (4,5,5,sub_grid,sub_grid)
                a_components_pad = np.zeros(
                    (4, 2 * nbfe + 1, 2 * nbfe + 1, bin_size * n_sub + 2 * nbfe, bin_size * m_sub + 2 * nbfe)
                )  # (4,5,5,sub_grid,sub_grid)

                for comp in range(4):
                    for j in range(2 * nbfe + 1):
                        for i in range(2 * nbfe + 1):
                            tmp = (
                                t.dot(
                                    a_components[
                                        comp,
                                        j,
                                        i,
                                        gj * n_sub : (gj + 1) * n_sub,
                                        gi * m_sub : (gi + 1) * m_sub,
                                    ]
                                )
                            ).dot(
                                t.T
                            )  # sub_grid*sub_grid
                            a_components_pad[comp, j, i, :, :] = np.pad(
                                tmp, [(nbfe, nbfe), (nbfe, nbfe)], mode="symmetric"
                            )

                # convolve aX_ij with Q_ij
                for comp in range(4):
                    for dy in range(-nbfe, nbfe + 1):
                        for dx in range(-nbfe, nbfe + 1):
                            dQ_components[
                                comp,
                                gj * bin_size * n_sub : (gj + 1) * bin_size * n_sub,
                                gi * bin_size * m_sub : (gi + 1) * bin_size * m_sub,
                            ] += (
                                a_components_pad[
                                    comp,
                                    nbfe + dy,
                                    nbfe + dx,
                                    nbfe - dy : nbfe - dy + bin_size * n_sub,
                                    nbfe - dx : nbfe - dx + bin_size * m_sub,
                                ]
                                * array_pad[
                                    -dy
                                    + nbfe
                                    + gj * bin_size * n_sub : -dy
                                    + nbfe
                                    + (gj + 1) * bin_size * n_sub,
                                    -dx
                                    + nbfe
                                    + gi * bin_size * m_sub : -dx
                                    + nbfe
                                    + (gi + 1) * bin_size * m_sub,
                                ]
                            )

                    dj = int(np.sin(comp * np.pi / 2))
                    di = int(np.cos(comp * np.pi / 2))

                    dQ_components[
                        comp,
                        gj * bin_size * n_sub : (gj + 1) * bin_size * n_sub,
                        gi * bin_size * m_sub : (gi + 1) * bin_size * m_sub,
                    ] *= 0.5 * (
                        array_pad[
                            nbfe + gj * bin_size * n_sub : nbfe + (gj + 1) * bin_size * n_sub,
                            nbfe + gi * bin_size * m_sub : nbfe + (gi + 1) * bin_size * m_sub,
                        ]
                        + array_pad[
                            dj + nbfe + gj * bin_size * n_sub : dj + nbfe + (gj + 1) * bin_size * n_sub,
                            di + nbfe + gi * bin_size * m_sub : di + nbfe + (gi + 1) * bin_size * m_sub,
                        ]
                    )
                    dQ_components[
                        comp,
                        gj * bin_size * n_sub : (gj + 1) * bin_size * n_sub,
                        gi * bin_size * m_sub : (gi + 1) * bin_size * m_sub,
                    ] *= 0.5 * (
                        array_pad[
                            nbfe + gj * bin_size * n_sub : nbfe + (gj + 1) * bin_size * n_sub,
                            nbfe + gi * bin_size * m_sub : nbfe + (gi + 1) * bin_size * m_sub,
                        ]
                        + array_pad[
                            dj + nbfe + gj * bin_size * n_sub : dj + nbfe + (gj + 1) * bin_size * n_sub,
                            di + nbfe + gi * bin_size * m_sub : di + nbfe + (gi + 1) * bin_size * m_sub,
                        ]
                    )

        im.array[:, :] -= dQ_components.sum(axis=0)
        im.array[:, 1:] += dQ_components[0][:, :-1]
        im.array[1:, :] += dQ_components[1][:-1, :]
        im.array[:, :-1] += dQ_components[2][:, 1:]
        im.array[:-1, :] += dQ_components[3][1:, :]

        return im

    def get_eff_sky_bg(self, pointing, radec):
        """
        Calculate effective sky background per pixel for nominal roman pixel scale.

        Input
        pointing            : Pointing object
        radec               : World coordinate position of image
        """

        sky_level = roman.getSkyLevel(pointing.bpass, world_pos=radec, date=pointing.date)
        sky_level *= (1.0 + roman.stray_light_fraction) * roman.pixel_scale**2

        return sky_level

    def translate_cvz(self, orig_radec, field_ra=9.5, field_dec=-44, cvz_ra=61.24, cvz_dec=-48.42):

        ra = orig_radec.ra / galsim.degrees - field_ra
        dec = orig_radec.dec / galsim.degrees - field_dec
        ra += cvz_ra / np.cos(cvz_dec * np.pi / 180)
        dec += cvz_dec
        return galsim.CelestialCoord(ra * galsim.degrees, dec * galsim.degrees)

    def setup_sky(self, im, pointing, rng, rng_iter, force_cvz=False):
        """
        Setup sky

        First we get the amount of zodaical light for a position corresponding to the position of
        the object. The results are provided in units of e-/arcsec^2, using the default Roman
        exposure time since we did not explicitly specify one. Then we multiply this by a factor
        >1 to account for the amount of stray light that is expected. If we do not provide a date
        for the observation, then it will assume that it's the vernal equinox (sun at (0,0) in
        ecliptic coordinates) in 2025.

        Input
        im                  : Image
        pointing            : Pointing object
        radec               : World coordinate position of image
        local_wcs           : Local WCS
        """

        self.rng = rng
        self.noise = galsim.PoissonNoise(self.rng)
        self.rng_np = np.random.default_rng(rng_iter)
        if self.df is None:
            self.dark_current_ = roman.dark_current * roman.exptime
        else:
            self.dark_current_ = (
                roman.dark_current * roman.exptime + self.df["DARK"][:, :].flatten() * roman.exptime
            )
        if self.df is None:
            self.gain = roman.gain
        else:
            self.gain = self.df["GAIN"][:, :]
        self.read_noise = galsim.GaussianNoise(self.rng, sigma=roman.read_noise)

        # Build current specification sky level if sky level not given
        if force_cvz:
            radec = self.translate_cvz(pointing.radec)
        else:
            radec = pointing.radec
        sky_level = roman.getSkyLevel(pointing.bpass, world_pos=radec, date=pointing.date)
        sky_level *= 1.0 + roman.stray_light_fraction
        # Make a image of the sky that takes into account the spatially variable pixel scale. Note
        # that makeSkyImage() takes a bit of time. If you do not care about the variable pixel
        # scale, you could simply compute an approximate sky level in e-/pix by multiplying
        # sky_level by roman.pixel_scale**2, and add that to final_image.

        # Create sky image
        self.sky = galsim.Image(bounds=im.bounds, wcs=pointing.WCS)
        pointing.WCS.makeSkyImage(self.sky, sky_level)

        # This image is in units of e-/pix. Finally we add the expected thermal backgrounds in this
        # band. These are provided in e-/pix/s, so we have to multiply by the exposure time.
        self.sky += roman.thermal_backgrounds[pointing.filter] * roman.exptime

        # Median of dark current is used here instead of mean since hot pixels contribute significantly to
        # the mean. Stastistics of dark current for the current test detector file: (mean, std, median, max)
        # ~ (35, 3050, 0.008, 1.2E6)  (e-/p). Hot pixels could be removed in further analysis using the
        # dq array.
        self.sky_mean = np.mean(
            np.round((np.round(self.sky.array) + round(np.median(self.dark_current_))) / np.mean(self.gain))
        )

        self.sky.addNoise(self.noise)

    def add_background(self, im):
        """
        Add backgrounds to image (sky, thermal).

        First we get the amount of zodaical light for a position corresponding to the position of
        the object. The results are provided in units of e-/arcsec^2, using the default Roman
        exposure time since we did not explicitly specify one. Then we multiply this by a factor
        >1 to account for the amount of stray light that is expected. If we do not provide a date
        for the observation, then it will assume that it's the vernal equinox (sun at (0,0) in
        ecliptic coordinates) in 2025.

        Input
        im                  : Image
        """

        # If requested, dump an initial fits image to disk for diagnostics
        if self.params["save_diff"]:
            orig = im.copy()
            orig.write("orig.fits")

        # Adding sky level to the image.
        im += self.sky[self.sky.bounds & im.bounds]

        # If requested, dump a post-change fits image to disk for diagnostics
        if self.params["save_diff"]:
            prev = im.copy()
            diff = prev - orig
            diff.write("sky_a.fits")

        return im

    def recip_failure(self, im, exptime=roman.exptime, alpha=roman.reciprocity_alpha, base_flux=1.0):
        """
        Introduce reciprocity failure to image.

        Reciprocity, in the context of photography, is the inverse relationship between the
        incident flux (I) of a source object and the exposure time (t) required to produce a given
        response(p) in the detector, i.e., p = I*t. However, in NIR detectors, this relation does
        not hold always. The pixel response to a high flux is larger than its response to a low
        flux. This flux-dependent non-linearity is known as 'reciprocity failure', and the
        approximate amount of reciprocity failure for the Roman detectors is known, so we can
        include this detector effect in our images.

        Input
        im        : image
        exptime   : Exposure time
        alpha     : Reciprocity alpha
        base_flux : Base flux
        """

        # Add reciprocity effect
        im.addReciprocityFailure(exp_time=exptime, alpha=alpha, base_flux=base_flux)

        return im

    def dark_current(self, im):
        """
        Adding dark current to the image.

        Even when the detector is unexposed to any radiation, the electron-hole pairs that
        are generated within the depletion region due to finite temperature are swept by the
        high electric field at the junction of the photodiode. This small reverse bias
        leakage current is referred to as 'dark current'. It is specified by the average
        number of electrons reaching the detectors per unit time and has an associated
        Poisson noise since it is a random event.

        Input
        im           : image
        """

        if self.df is None:
            self.im_dark = im.copy()
            dark_current_ = self.dark_current_
            dark_noise = galsim.DeviateNoise(galsim.PoissonDeviate(self.rng, dark_current_))
            im.addNoise(dark_noise)
            self.im_dark = im - self.im_dark

        else:

            dark_current_ = self.dark_current_.clip(0)

            # opt for numpy random geneator instead for speed
            self.im_dark = self.rng_np.poisson(dark_current_).reshape(im.array.shape).astype(im.dtype)
            im.array[:, :] += self.im_dark

        # NOTE: Sky level and dark current might appear like a constant background that can be
        # simply subtracted. However, these contribute to the shot noise and matter for the
        # non-linear effects that follow. Hence, these must be included at this stage of the
        # image generation process. We subtract these backgrounds in the end.

        return im

    def saturate(self, im, saturation=100000):
        """
        Clip the saturation level
        Input
        im                     : image
        SATURATE[4096,4096]    : saturation map
        """

        if self.df is None:
            saturation_array = np.ones_like(im.array) * saturation
        else:
            saturation_array = self.df["SATURATE"][:, :]  # 4096x4096 array
        where_sat = np.where(im.array > saturation_array)
        im.array[where_sat] = saturation_array[where_sat]

        return im

    def deadpix(self, im):
        """
        Apply dead pixel mask
        Input
        im                   : image
        BADPIX[4096,4096]    : bit mask with the first bit flags dead pixels
        """

        dead_mask = self.df["BADPIX"][:, :] & 1  # 4096x4096 array
        im.array[dead_mask > 0] = 0

        return im

    def vtpe(self, im):
        """
        Apply vertical trailing pixel effect.
        The vertical trailing pixel effect (VTPE) is a non-linear effect that is
        related to readout patterns.
        Q'[j,i] = Q[j,i] + f(  Q[j,i] - Q[j-1, i]  ),
        where f( dQ ) = dQ ( a + b * ln(1 + |dQ|/dQ0) )
        Input
        im           : image
        VTPE[0,512,512]  : coefficient a binned in 8x8
        VTPE[1,512,512]  : coefficient a
        VTPE[2,512,512]  : coefficient dQ0
        """

        # expand 512x512 arrays to 4096x4096

        t = np.zeros((4096, 512))
        for row in range(t.shape[0]):
            t[row, row // 8] = 1
        a_vtpe = t.dot(self.df["VTPE"][0, :, :][0]).dot(t.T)
        # NaN check
        if np.isnan(a_vtpe).any():
            print("vtpe skipped due to NaN in file")
            return im
        b_vtpe = t.dot(self.df["VTPE"][1, :, :][0]).dot(t.T)
        dQ0 = t.dot(self.df["VTPE"][2, :, :][0]).dot(t.T)

        dQ = im.array - np.roll(im.array, 1, axis=0)
        dQ[0, :] *= 0

        im.array[:, :] += dQ * (a_vtpe + b_vtpe * np.log(1.0 + np.abs(dQ) / dQ0))
        return im

    def add_persistence(self, im, pointing):
        """
        Applying the persistence effect.

        Even after reset, some charges from prior illuminations are trapped in defects of semiconductors.
        Trapped charges are gradually released and generate the flux-dependent persistence signal.
        Here we adopt the same fermi-linear model to describe the illumination dependence and time dependence
        of the persistence effect for all SCAs.

        Input
        im                    : image
        pointing              : pointing object
        """

        # load the dithers of sky images that were simulated
        dither_sca_array = np.loadtxt(self.params["dither_from_file"]).astype(int)

        # select adjacent exposures for the same sca (within 10*roman.exptime)
        dither_list_selected = dither_sca_array[dither_sca_array[:, 1] == pointing.sca, 0]
        dither_list_selected = dither_list_selected[np.abs(dither_list_selected - pointing.visit) < 10]
        p_list = np.array([get_pointing(self.params, i, pointing.sca) for i in dither_list_selected])
        dt_list = np.array([(pointing.date - p.date).total_seconds() for p in p_list])
        p_pers = p_list[np.where((dt_list > 0) & (dt_list < roman.exptime * 10))]

        if self.df is None:
            # iterate over previous exposures
            for p in p_pers:
                dt = (
                    pointing.date - p.date
                ).total_seconds() - roman.exptime / 2  # avg time since end of exposures
                self.params["output"]["file_name"]["items"] = [p.filter, p.visit, p.sca]
                imfilename = ParseValue(self.params["output"], "file_name", self.params, str)[0]
                fn = os.path.join(self.params["output"]["dir"], imfilename)

                # apply all the effects that occured before persistence on the previouse exposures
                # since max of the sky background is of order 100, it is thus negligible for persistence
                bound_pad = galsim.BoundsI(xmin=1, ymin=1, xmax=4088, ymax=4088)
                x = galsim.Image(bound_pad)
                x.array[:, :] = galsim.Image(fio.FITS(fn)[0].read()).array[:, :]
                x = self.recip_failure(x)

                x = x.clip(0)  # remove negative stimulus

                im.array[:, :] += galsim.roman.roman_detectors.fermi_linear(x.array, dt) * roman.exptime

        else:

            # setup parameters for persistence
            Q01 = self.df["PERSIST"].read_header()["Q01"]
            Q02 = self.df["PERSIST"].read_header()["Q02"]
            Q03 = self.df["PERSIST"].read_header()["Q03"]
            Q04 = self.df["PERSIST"].read_header()["Q04"]
            Q05 = self.df["PERSIST"].read_header()["Q05"]
            Q06 = self.df["PERSIST"].read_header()["Q06"]
            alpha = self.df["PERSIST"].read_header()["ALPHA"]

            # iterate over previous exposures
            for p in p_pers:
                dt = (
                    pointing.date - p.date
                ).total_seconds() - roman.exptime / 2  # avg time since end of exposures
                fac_dt = (
                    roman.exptime / 2.0
                ) / dt  # linear time dependence (approximate until we get t1 and Delat t of the data)
                self.params["output"]["file_name"]["items"] = [p.filter, p.visit, p.sca]
                imfilename = ParseValue(self.params["output"], "file_name", self.params, str)[0]
                fn = os.path.join(self.params["output"]["dir"], imfilename)

                # apply all the effects that occured before persistence on the previouse exposures
                # since max of the sky background is of order 100, it is thus negligible for persistence
                # same for brighter fatter effect
                bound_pad = galsim.BoundsI(xmin=1, ymin=1, xmax=4096, ymax=4096)
                x = galsim.Image(bound_pad)
                x.array[4:-4, 4:-4] = galsim.Image(fio.FITS(fn)[0].read()).array[:, :]
                x = self.qe(x).array[:, :]

                x = x.clip(0.1)  # remove negative and zero stimulus

                # Do linear interpolation
                a = np.zeros(x.shape)
                a += ((x < Q01)) * x / Q01
                a += ((x >= Q01) & (x < Q02)) * (Q02 - x) / (Q02 - Q01)
                im.array[:, :] += a * self.df["PERSIST"][0, :, :][0] * fac_dt

                a = np.zeros(x.shape)
                a += ((x >= Q01) & (x < Q02)) * (x - Q01) / (Q02 - Q01)
                a += ((x >= Q02) & (x < Q03)) * (Q03 - x) / (Q03 - Q02)
                im.array[:, :] += a * self.df["PERSIST"][1, :, :][0] * fac_dt

                a = np.zeros(x.shape)
                a += ((x >= Q02) & (x < Q03)) * (x - Q02) / (Q03 - Q02)
                a += ((x >= Q03) & (x < Q04)) * (Q04 - x) / (Q04 - Q03)
                im.array[:, :] += a * self.df["PERSIST"][2, :, :][0] * fac_dt

                a = np.zeros(x.shape)
                a += ((x >= Q03) & (x < Q04)) * (x - Q03) / (Q04 - Q03)
                a += ((x >= Q04) & (x < Q05)) * (Q05 - x) / (Q05 - Q04)
                im.array[:, :] += a * self.df["PERSIST"][3, :, :][0] * fac_dt

                a = np.zeros(x.shape)
                a += ((x >= Q04) & (x < Q05)) * (x - Q04) / (Q05 - Q04)
                a += ((x >= Q05) & (x < Q06)) * (Q06 - x) / (Q06 - Q05)
                im.array[:, :] += a * self.df["PERSIST"][4, :, :][0] * fac_dt

                a = np.zeros(x.shape)
                a += ((x >= Q05) & (x < Q06)) * (x - Q05) / (Q06 - Q05)
                a += ((x >= Q06)) * (x / Q06) ** alpha  # avoid fractional power of negative values
                im.array[:, :] += a * self.df["PERSIST"][5, :, :][0] * fac_dt

        return im

    def nonlinearity(self, im, NLfunc=roman.NLfunc):
        """
        Applying a quadratic non-linearity.

        Note that users who wish to apply some other nonlinearity function (perhaps for other NIR
        detectors, or for CCDs) can use the more general nonlinearity routine, which uses the
        following syntax:
        final_image.applyNonlinearity(NLfunc=NLfunc)
        with NLfunc being a callable function that specifies how the output image pixel values
        should relate to the input ones.

        Input
        im     : Image
        NLfunc : Nonlinearity function
        """

        # Apply the Roman nonlinearity routine, which knows all about the nonlinearity expected in
        # the Roman detectors. Alternately, use a user-provided function.
        if self.df is None:
            im.applyNonlinearity(NLfunc=NLfunc)
        else:
            im.array[:, :] -= (
                self.df["CNL"][0, :, :][0] * im.array**2
                + self.df["CNL"][1, :, :][0] * im.array**3
                + self.df["CNL"][2, :, :][0] * im.array**4
            )

        return im

    def interpix_cap(self, im, kernel=roman.ipc_kernel):
        """
        Including Interpixel capacitance

        The voltage read at a given pixel location is influenced by the charges present in the
        neighboring pixel locations due to capacitive coupling of sense nodes. This interpixel
        capacitance effect is modeled as a linear effect that is described as a convolution of a
        3x3 kernel with the image. The Roman IPC routine knows about the kernel already, so the
        user does not have to supply it.

        Input
        im      : image
        kernel  : Interpixel capacitance kernel
        """

        # Apply interpixel capacitance
        if self.df is None:
            im.applyIPC(kernel, edge_treatment="extend", fill_value=None)
        else:
            # pad the array by one pixel at the four edges
            num_grids = 4  # num_grids <= 8
            grid_size = 4096 // num_grids

            array_pad = im.array[4:-4, 4:-4]  # it's an array instead of img
            array_pad = np.pad(array_pad, [(5, 5), (5, 5)], mode="symmetric")  # 4098x4098 array

            K = self.df["IPC"][:, :, :, :]  # 3,3,512, 512

            t = np.zeros((grid_size, 512))
            for row in range(t.shape[0]):
                t[row, row // (grid_size // 512)] = 1

            array_out = np.zeros((4096, 4096))
            # split job in sub_grids to reduce memory
            for gj in range(num_grids):
                for gi in range(num_grids):
                    K_pad = np.zeros((3, 3, grid_size + 2, grid_size + 2))

                    for j in range(3):
                        for i in range(3):
                            tmp = (t.dot(K[j, i, :, :])).dot(t.T)  # grid_sizexgrid_size
                            K_pad[j, i, :, :] = np.pad(tmp, [(1, 1), (1, 1)], mode="symmetric")

                    for dy in range(-1, 2):
                        for dx in range(-1, 2):

                            array_out[
                                gj * grid_size : (gj + 1) * grid_size,
                                gi * grid_size : (gi + 1) * grid_size,
                            ] += (
                                K_pad[
                                    1 + dy,
                                    1 + dx,
                                    1 - dy : 1 - dy + grid_size,
                                    1 - dx : 1 - dx + grid_size,
                                ]
                                * array_pad[
                                    1 - dy + gj * grid_size : 1 - dy + (gj + 1) * grid_size,
                                    1 - dx + gi * grid_size : 1 - dx + (gi + 1) * grid_size,
                                ]
                            )

            im.array[:, :] = array_out
        return im

    def add_read_noise(self, im):
        """
        Adding read noise

        Read noise is the noise due to the on-chip amplifier that converts the charge into an
        analog voltage.  We already applied the Poisson noise due to the sky level, so read noise
        should just be added as Gaussian noise

        Input
        im    : image
        """

        # Create noise realisation and apply it to image
        if self.df is None:
            self.im_read = im.copy()
            im.addNoise(self.read_noise)
            self.im_read = im - self.im_read
            # self.sky.addNoise(self.read_noise)
        else:
            # use numpy random generator to draw 2-d noise map
            read_noise = self.df["READ"][2, :, :].flatten()  # flattened 4096x4096 array
            self.im_read = (
                self.rng_np.normal(loc=0.0, scale=read_noise).reshape(im.array.shape).astype(im.dtype)
            )
            im.array[:, :] += self.im_read

            # noise_array = self.rng_np.normal(loc=0., scale=read_noise)
            # 4088x4088 img
            # self.sky.array[:,:] += noise_array.reshape(im.array.shape)[4:-4, 4:-4].astype(self.sky.dtype)

        return im

    def e_to_ADU(self, im):
        """
        We divide by the gain to convert from e- to ADU. Currently, the gain value in the Roman
        module is just set to 1, since we don't know what the exact gain will be, although it is
        expected to be approximately 1. Eventually, this may change when the camera is assembled,
        and there may be a different value for each SCA. For now, there is just a single number,
        which is equal to 1.

        Input
        im : image
        """
        if self.df is None:
            return im / roman.gain
        else:
            bias = self.df["BIAS"][:, :]  # 4096x4096 img
            t = np.zeros((4096, 32))
            for row in range(t.shape[0]):
                t[row, row // 128] = 1
            gain_expand = (t.dot(self.gain)).dot(t.T)  # 4096x4096 gain img
            im.array[:, :] = im.array / gain_expand + bias
            return im

    def add_gain(self, im):
        """
        We divide by the gain to convert from e- to ADU.
        Input
        im : image
        GAIN : 32x32 float img in unit of e-/adu, mean(GAIN)~ 1.6
        """

        gain = self.df["GAIN"][:, :]  # 32x32 img

        t = np.zeros((4096, 32))
        for row in range(t.shape[0]):
            t[row, row // 128] = 1
        gain_expand = (t.dot(gain)).dot(t.T)  # 4096x4096 gain img
        im.array[:, :] /= gain_expand
        return im

    def add_bias(self, im):
        """
        Add the voltage bias.
        Input
        im : image
        BIAS : 4096x4096 uint16 bias img (in unit of DN), mean(bias) ~ 6.7k
        """

        bias = self.df["BIAS"][:, :]  # 4096x4096 img

        im.array[:, :] += bias
        return im

    def finalize_sky_im(self, im, pointing):
        """
        Finalize sky background for subtraction from final image. Add dark current,
        convert to analog voltage, and quantize.

        Input
        im : sky image
        """

        if self.df is None:
            im.quantize()
            im += self.im_dark
            im = self.saturate(im)
            im += self.im_read
            im = self.e_to_ADU(im)
            im.quantize()
        else:

            bound_pad = galsim.BoundsI(xmin=1, ymin=1, xmax=4096, ymax=4096)
            im_pad = galsim.Image(bound_pad)
            im_pad.array[4:-4, 4:-4] = im.array[:, :]

            im_pad = self.qe(im_pad)
            im_pad = self.bfe(im_pad)
            im_pad = self.add_persistence(im_pad, pointing)
            im_pad.quantize()
            im_pad += self.im_dark
            im_pad = self.saturate(im_pad)
            im_pad = self.nonlinearity(im_pad)
            im_pad = self.interpix_cap(im_pad)
            im_pad = self.deadpix(im_pad)
            im_pad = self.vtpe(im_pad)
            im_pad += self.im_read
            im_pad = self.add_gain(im_pad)
            im_pad = self.add_bias(im_pad)
            im_pad.quantize()
            # output 4088x4088 img in uint16
            im.array[:, :] = im_pad.array[4:-4, 4:-4]
            im = galsim.Image(im, dtype=np.uint16)

        return im
