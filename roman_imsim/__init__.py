from importlib.metadata import version, PackageNotFoundError

try:
    from lsst.utils.threads import disable_implicit_threading

    disable_implicit_threading()
except ImportError:
    pass

try:
    __version__ = version("roman_imsim")
except PackageNotFoundError:
    pass

from .bandpass import *
from .detector_physics import *

# Import core modules for public use
from .noise import *
from .obseq import *
from .output_asdf import *
from .photonOps import *
from .psf import *
from .sca import *
from .skycat import *
from .stamp import *
from .wcs import *
