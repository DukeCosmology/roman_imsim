import sys

import roman_imsim

params = sys.argv[1]
visit = int(sys.argv[2])
sca = int(sys.argv[3])
if len(sys.argv) > 4:
    dither_from_file = sys.argv[4]
else:
    import warnings
    warnings.warn("Ditherlist not provided. Persistence effect will not simulated.")
    dither_from_file = None

if len(sys.argv)>5:
    sca_path = sys.argv[5]
else:
    sca_path = None

roman_imsim.detector_physics.modify_image(params,visit,sca,dither_from_file,sca_filepath=sca_path)
