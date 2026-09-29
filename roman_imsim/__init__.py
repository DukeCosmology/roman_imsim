from importlib.metadata import PackageNotFoundError, version

try:
    from lsst.utils.threads import disable_implicit_threading

    disable_implicit_threading()
except ImportError:
    pass

try:
    __version__ = version("roman_imsim")
except PackageNotFoundError:
    pass

# Register the template on importing
from ._templates import *

# Import core modules for public use
from .noise import *
from .obseq import *
from .photonOps import *
from .psf import *
from .sca import *
from .skycat import *
from .stamp import *
from .wcs import *
