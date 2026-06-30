"""Standalone script to convert existing FITS files to ASDF."""

import galsim
from roman_imsim.output_asdf import RomanASDFBuilder
import logging

logger = logging.getLogger()

builder = RomanASDFBuilder()
builder.include_raw_header = False

fname_path = (
    "/Users/ajk107/code/roman/roman_imsim/roman_imsim_testdata/"
    "output/RomanTDS_prism/images/simple_model/"
    "Roman_TDS_simple_model_test_SNPrism_4191_4.fits"
)
visit = 4191
base = {"input": {"obseq_data": {"visit": visit}}}
config = {}
im = galsim.fits.read(fname_path, hdu=1, read_header=True)
im.header["FILTER"] = "PRISM"

builder._writeASDF(config, base, im, fname_path.replace(".fits", ".asdf"), logger)
