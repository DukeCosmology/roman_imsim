"""Standalone script to convert existing FITS file to ASDF.
"""
import galsim
import os
from roman_imsim.output_asdf import RomanASDFBuilder
import logging

logger = logging.getLogger()

builder = RomanASDFBuilder()
builder.include_raw_header = False

dir_path = "/hpc/group/cosmology/ajk107/code/roman_imsim/RomanTDS_prism/images/simple_model"
names = os.listdir(dir_path)
for name in names:
    if name.endswith(".fits.gz"):
        fname_path = os.path.join(dir_path, name)
        visit = int(name.split("_")[-2])
        base = {"input": {"obseq_data": {"visit": visit}}}
        config = {}
        im = galsim.fits.read(fname_path, hdu=1, read_header=True)
        im.header["FILTER"] = "PRISM"

        builder._writeASDF(config, base, im, fname_path.replace(".fits.gz", ".asdf"), logger)  
        logger.info(f"Converted {name}")
