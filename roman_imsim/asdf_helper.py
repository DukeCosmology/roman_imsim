import warnings

import numpy as np
from astropy import time

from roman_datamodels import stnode

import sys
sys.path.insert(0, "/hpc/group/cosmology/boyan/roman_datamodels/src/roman_datamodels/")
from _maker_utils._base import MESSAGE, save_node
from _maker_utils._common_meta import (
    mk_catalog_meta,
    mk_common_meta,
    mk_guidewindow_meta,
    mk_l1_detector_guidewindow_meta,
    mk_l1_face_guidewindow_meta,
    mk_l2_meta,
    mk_mosaic_catalog_meta,
    mk_msos_stack_meta,
    mk_ramp_meta,
    mk_wcs,
    mk_wfi_wcs_common_meta,
)



def mk_level2_image_with_wcs(*, shape=(4088, 4088), n_groups=8, filepath=None, wcs = None, **kwargs):
    """
    Create a dummy level 2 Image instance (or file) with arrays and valid values
    for attributes required by the schema.

    Parameters
    ----------
    shape : tuple, int
        (optional, keyword-only) Shape (y, x) of data array in the model (and
        its corresponding dq/err arrays). This specified size does NOT include
        the four-pixel border of reference pixels - those are trimmed at level
        2.  This size, however, is used to construct the additional arrays that
        contain the original border reference pixels (i.e if shape = (10, 10),
        the border reference pixel arrays will have (y, x) dimensions (14, 4)
        and (4, 14)). Default is 4088 x 4088.
        If shape is a tuple of length 3, the first element is assumed to be the
        n_groups and will override any settings there.

    n_groups : int
        (optional, keyword-only) The level 2 file is flattened, but it contains
        arrays for the original reference pixels which remain 3D. n_groups
        specifies what the z dimension of these arrays should be. Defaults to 8.

    filepath : str
        (optional, keyword-only) File name and path to write model to.

    Returns
    -------
    roman_datamodels.stnode.WfiImage
    """
    if len(shape) > 2:
        n_groups = shape[0]
        shape = shape[1:3]

        warnings.warn(
            f"{MESSAGE} assuming the first entry is n_groups followed by y, x. The remaining is thrown out!",
            UserWarning,
            stacklevel=2,
        )

    wfi_image = stnode.WfiImage()

    wfi_image["meta"] = mk_l2_meta(**kwargs.get("meta", {}))

    # add border reference pixel arrays
    wfi_image["border_ref_pix_left"] = kwargs.get("border_ref_pix_left", np.zeros((n_groups, shape[0] + 8, 4), dtype=np.float32))
    wfi_image["border_ref_pix_right"] = kwargs.get(
        "border_ref_pix_right", np.zeros((n_groups, shape[0] + 8, 4), dtype=np.float32)
    )
    wfi_image["border_ref_pix_top"] = kwargs.get("border_ref_pix_top", np.zeros((n_groups, shape[0] + 8, 4), dtype=np.float32))
    wfi_image["border_ref_pix_bottom"] = kwargs.get(
        "border_ref_pix_bottom", np.zeros((n_groups, shape[0] + 8, 4), dtype=np.float32)
    )

    # and their dq arrays
    wfi_image["dq_border_ref_pix_left"] = kwargs.get("dq_border_ref_pix_left", np.zeros((shape[0] + 8, 4), dtype=np.uint32))
    wfi_image["dq_border_ref_pix_right"] = kwargs.get("dq_border_ref_pix_right", np.zeros((shape[0] + 8, 4), dtype=np.uint32))
    wfi_image["dq_border_ref_pix_top"] = kwargs.get("dq_border_ref_pix_top", np.zeros((4, shape[1] + 8), dtype=np.uint32))
    wfi_image["dq_border_ref_pix_bottom"] = kwargs.get("dq_border_ref_pix_bottom", np.zeros((4, shape[1] + 8), dtype=np.uint32))

    # add amp 33 ref pixel array
    amp33_size = (n_groups, 4096, 128)
    wfi_image["amp33"] = kwargs.get("amp33", np.zeros(amp33_size, dtype=np.uint16))
    wfi_image["data"] = kwargs.get("data", np.zeros(shape, dtype=np.float32))
    wfi_image["dq"] = kwargs.get("dq", np.zeros(shape, dtype=np.uint32))
    wfi_image["err"] = kwargs.get("err", np.zeros(shape, dtype=np.float32))

    wfi_image["var_poisson"] = kwargs.get("var_poisson", np.zeros(shape, dtype=np.float32))
    wfi_image["var_rnoise"] = kwargs.get("var_rnoise", np.zeros(shape, dtype=np.float32))
    wfi_image["var_flat"] = kwargs.get("var_flat", np.zeros(shape, dtype=np.float32))
    # wfi_image["cal_logs"] = mk_cal_logs(**kwargs)

    wfi_image["meta"]["wcs"] = kwargs.get("wcs", mk_wcs())

    return save_node(wfi_image, filepath=filepath)
