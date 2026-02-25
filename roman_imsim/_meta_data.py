import os

imsim_dir = os.path.split(os.path.realpath(__file__))[0]

if "ROMAN_IMSIM_TESTDATA_DIR" in os.environ:  # pragma: no cover
    data_dir = os.environ["ROMAN_IMSIM_TESTDATA_DIR"]
else:
    data_dir = "_".join([imsim_dir, "testdata"])

config_dir = os.path.join(imsim_dir, "config")
