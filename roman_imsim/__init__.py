try:
    from lsst.utils.threads import disable_implicit_threading
    disable_implicit_threading()
except:
    pass
from .obseq import *
from .psf import *
from .sca import *
from .stamp import *
from .wcs import *
from .optical_model_utils import *
from .skycat import *
from .photonOps import *
from .bandpass import *
from .detector_physics import *
#from . import photonOps  # this registers SlitlessSpec
