from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.io import ascii

REFERENCE_DIR = "reference"
OUTPUT_DIR = "output"

HACK_REPO = "SimHackJune2025"


pytestmark = pytest.mark.skipif(
    not Path(HACK_REPO).exists(),
    reason=f"Test data not available; {HACK_REPO} not found",
)

ATOL = 0
RTOL = 1e-6


def test_compare_truth():
    reference_truth = (
        Path(HACK_REPO) / REFERENCE_DIR / "RomanWAS_new" / "truth" / "Roman_WAS_index_J129_12909_4.txt"
    )
    output_truth = (
        Path(HACK_REPO) / OUTPUT_DIR / "RomanWAS_new" / "truth" / "Roman_WAS_index_J129_12909_4.txt"
    )

    output_table = ascii.read(output_truth)
    reference_table = ascii.read(reference_truth)

    np.testing.assert_equal(reference_table.colnames, output_table.colnames)
    np.testing.assert_equal(len(reference_table), len(output_table))

    for output_col, reference_col in zip(
        output_table.itercols(),
        reference_table.itercols(),
    ):
        if isinstance(reference_col.dtype, np.dtypes.StrDType):
            np.testing.assert_equal(output_col, reference_col)
        else:
            np.testing.assert_allclose(output_col, reference_col, rtol=RTOL, atol=ATOL)


def test_compare_image():
    reference_image = (
        Path(HACK_REPO)
        / REFERENCE_DIR
        / "RomanWAS_new"
        / "images"
        / "truth"
        / "Roman_WAS_truth_J129_12909_4.fits.gz"
    )
    output_image = (
        Path(HACK_REPO)
        / OUTPUT_DIR
        / "RomanWAS_new"
        / "images"
        / "truth"
        / "Roman_WAS_truth_J129_12909_4.fits.gz"
    )

    output_hdul = fits.open(output_image)
    reference_hdul = fits.open(reference_image)

    output_data = output_hdul[0].data
    reference_data = reference_hdul[0].data

    np.testing.assert_allclose(output_data, reference_data, rtol=RTOL, atol=ATOL)
