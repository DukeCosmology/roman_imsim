try:
    from lsst.utils.threads import disable_implicit_threading

    disable_implicit_threading()
except ImportError:
    pass

# Register the template on importing
from ._templates import *
from .bandpass import *
from .detector_physics import *

# Import core modules for public use
from .obseq import *
from .photonOps import *
from .psf import *
from .sca import *
from .skycat import *
from .stamp import *
from .wcs import *
