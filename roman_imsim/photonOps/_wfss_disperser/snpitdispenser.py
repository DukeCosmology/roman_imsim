import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import yaml

"""
Classes and numba-accelerated functions to implement the optical
model provided by the SSC (IPAC).

Written by R. Ryan (STScI)

Feb 8, 2026
"""


@nb.njit
def horner1(c, z):
    """
    Evaluate a 1D polynomial at z using Horner's method.

    Parameters
    ----------
    c : ndarray, shape (n,)
        Polynomial coefficients ordered from lowest to highest degree.
    z : float or ndarray
        Evaluation point(s).

    Returns
    -------
    p : float or ndarray
        Polynomial value at z.
    """
    nz = c.shape[0]
    p = c[-1]
    for i in range(nz-2, -1, -1):
        p *= z
        p += c[i]
    return p


@nb.njit
def horner2(c, y, z):
    """
    Evaluate a 2D polynomial at (y, z) using nested Horner's method.

    Parameters
    ----------
    c : ndarray, shape (ny, nz)
        Polynomial coefficients, where c[i, j] corresponds to y^i z^j.
    y, z : float or ndarray
        Evaluation point(s).

    Returns
    -------
    p : float or ndarray
        Polynomial value at (y, z).
    """
    ny = c.shape[0]
    p = horner1(c[-1], z)
    for i in range(ny-2, -1, -1):
        p *= y
        p += horner1(c[i], z)
    return p


@nb.njit
def horner3(c, x, y, z):
    """
    Evaluate a 3D polynomial at (x, y, z) using nested Horner's method.

    Parameters
    ----------
    c : ndarray, shape (nx, ny, nz)
        Polynomial coefficients, where c[i, j, k] corresponds to x^i y^j z^k.
    x, y, z : float or ndarray
        Evaluation point(s).

    Returns
    -------
    p : float or ndarray
        Polynomial value at (x, y, z).
    """
    nx = c.shape[0]
    p = horner2(c[-1], y, z)
    for i in range(nx-2, -1, -1):
        p *= x
        p += horner2(c[i], y, z)
    return p


@nb.njit(parallel=True)
def numba_disperse_pairwise(Xij, Yij, Cijk, Dijk, x, y, w):
    """
    Compute dispersed coordinates for pairwise inputs using Numba.

    This version assumes x, y, and w have identical shapes and computes
    one output coordinate pair per input triplet.

    Parameters
    ----------
    Xij, Yij : ndarray, shape (nx, ny)
        Polynomial coefficients mapping undispersed coordinates to
        Focal Plane Assembly (FPA) space.
    Cijk, Dijk : ndarray, shape (nw, nx, ny)
        Polynomial coefficients for dispersion in FPA space.
    x, y : ndarray
        Undispersed FPA coordinates.
    w : ndarray
        Transformed wavelength coordinate.

    Returns
    -------
    xt, yt : ndarray
        Dispersed FPA coordinates.
    """
    xt = np.empty_like(x)
    yt = np.empty_like(x)

    for idx in nb.prange(x.size):
        xx = x.flat[idx]
        yy = y.flat[idx]
        ww = w.flat[idx]

        dy = horner3(Cijk, ww, xx, yy)
        dx = horner3(Dijk, dy, xx, yy)

        xmpa = horner2(Xij, xx, yy)
        ympa = horner2(Yij, xx, yy)

        xt.flat[idx] = xmpa + dx
        yt.flat[idx] = ympa + dy
        
    return xt, yt


@nb.njit(parallel=True)
def numba_disperse(Xij, Yij, Cijk, Dijk, x, y, w):
    """
    Compute dispersed coordinates on a grid using Numba.

    This version assumes x and y are 1D arrays of positions, and w is a
    1D array of wavelengths. Outputs are 2D arrays indexed by
    (pixel_index, wavelength_index).

    Parameters
    ----------
    Xij, Yij : ndarray, shape (nx, ny)
        Polynomial coefficients mapping undispersed coordinates to
        Focal Plane Assembly (FPA) space.
    Cijk, Dijk : ndarray, shape (nw, nx, ny)
        Polynomial coefficients for dispersion in FPA space.
    x, y : ndarray, shape (npix,)
        Undispersed FPA coordinates.
    w : ndarray, shape (nlam,)
        Transformed wavelength coordinate.

    Returns
    -------
    xt, yt : ndarray, shape (npix, nlam)
        Dispersed FPA coordinates.
    """
    npix = x.size
    nlam = w.size

    dim = (npix, nlam)
    xt = np.empty(dim, dtype=float)
    yt = np.empty(dim, dtype=float)

    for idx in nb.prange(npix):
        xx = x.flat[idx]
        yy = y.flat[idx]

        xmpa = horner2(Xij, xx, yy)
        ympa = horner2(Yij, xx, yy)

        for jdx in nb.prange(nlam):
            ww = w.flat[jdx]

            dy = horner3(Cijk, ww, xx, yy)
            dx = horner3(Dijk, dy, xx, yy)

            xt[idx, jdx] = xmpa + dx
            yt[idx, jdx] = ympa + dy
    return xt, yt


@nb.njit(parallel=True)
def numba_deriv_pairwise(dCijk, Cijk, dDijk, x, y, w):
    """
    Compute derivatives of dispersed coordinates with respect to wavelength
    for pairwise inputs using Numba.

    Parameters
    ----------
    dCijk : ndarray, shape (nw-1, nx, ny)
        Derivative of Cijk coefficients with respect to the transformed
        wavelength coordinate.
    Cijk : ndarray, shape (nw, nx, ny)
        Original dispersion coefficients.
    dDijk : ndarray, shape (nw-1, nx, ny)
        Derivative of Dijk coefficients with respect to the dispersion
        coordinate.
    x, y : ndarray
        Undispersed FPA coordinates.
    w : ndarray
        Transformed wavelength coordinate.

    Returns
    -------
    dxdw, dydw : ndarray
        Derivatives of dispersed FPA coordinates with respect to transformed
        wavelength.
    """
    dxdw = np.empty_like(x)
    dydw = np.empty_like(x)
    for idx in nb.prange(x.size):
        xx = x.flat[idx]
        yy = y.flat[idx]
        ww = w.flat[idx]

        dy = horner3(Cijk, ww, xx, yy)

        dy_dw = horner3(dCijk, ww, xx, yy)
        dx_dy = horner3(dDijk, dy, xx, yy)
        
        dxdw[idx] = dy_dw * dx_dy
        dydw[idx] = dy_dw

    return dxdw, dydw    


@nb.njit(parallel=True)
def numba_deriv(dCijk, Cijk, dDijk, x, y, w):
    """
    Compute derivatives of dispersed coordinates with respect to wavelength
    on a grid using Numba.

    Parameters
    ----------
    dCijk : ndarray, shape (nw-1, nx, ny)
        Derivative of Cijk coefficients with respect to the transformed
        wavelength coordinate.
    Cijk : ndarray, shape (nw, nx, ny)
        Original dispersion coefficients.
    dDijk : ndarray, shape (nw-1, nx, ny)
        Derivative of Dijk coefficients with respect to the dispersion
        coordinate.
    x, y : ndarray, shape (npix,)
        Undispersed FPA coordinates.
    w : ndarray, shape (nlam,)
        Transformed wavelength coordinate.

    Returns
    -------
    dxdw, dydw : ndarray, shape (npix, nlam)
        Derivatives of dispersed FPA coordinates with respect to transformed
        wavelength.
    """
    npix = x.size
    nlam = w.size

    dim = (npix, nlam)
    dxdw = np.empty(dim, dtype=float)
    dydw = np.empty(dim, dtype=float)

    for idx in nb.prange(x.size):
        xx = x.flat[idx]
        yy = y.flat[idx]

        for jdx in nb.prange(nlam):
            ww = w.flat[jdx]

            dy = horner3(Cijk, ww, xx, yy)
            
            dy_dw = horner3(dCijk, ww, xx, yy)
            dx_dy = horner3(dDijk, dy, xx, yy)
            
            dxdw[idx, jdx] = dy_dw * dx_dy
            dydw[idx, jdx] = dy_dw

    return dxdw, dydw    


class LogTransformer:
    """
    Logarithmic wavelength transformer.

    Transforms physical wavelength values into a log-scaled coordinate
    relative to a reference wavelength, and provides inverse and derivative
    mappings.
    """
    ln10 = np.log(10.)

    def __init__(self, lam0):
        """
        Parameters
        ----------
        lam0 : float
            Reference wavelength.
        """
        self.lam0 = lam0

    def evaluate(self, lam):
        """
        Transform physical wavelength values into log-scaled coordinates.

        Parameters
        ----------
        lam : array_like
            Physical wavelengths.

        Returns
        -------
        w : ndarray
            Log-scaled wavelength coordinates.
        """
        return np.log10(lam/self.lam0)

    def invert(self, w):
        """
        Invert the log-scaled wavelength transformation.

        Parameters
        ----------
        w : array_like
            Log-scaled wavelength coordinates.

        Returns
        -------
        lam : ndarray
            Physical wavelengths.
        """
        return self.lam0 * (10.0 ** w)

    def deriv(self, lam):
        """
        Compute dλ/dw for the log-wavelength transform.

        Parameters
        ----------
        lam : array_like
            Physical wavelengths.

        Returns
        -------
        dldw : ndarray
            Derivative of wavelength with respect to the transformed coordinate.
        """
        return lam * self.ln10


class LinearTransformer:
    """
    Linear wavelength transformer.

    Transforms physical wavelength values into a linear coordinate
    relative to a reference wavelength, and provides inverse and derivative
    mappings.
    """

    def __init__(self, lam0):
        """
        Parameters
        ----------
        lam0 : float
            Reference wavelength.
        """
        self.lam0 = lam0

    def evaluate(self, lam):
        """
        Transform physical wavelength values into linear coordinates.

        Parameters
        ----------
        lam : array_like
            Physical wavelengths.

        Returns
        -------
        w : ndarray
            Linear wavelength coordinates.
        """
        return lam - self.lam0

    def invert(self, w):
        """
        Invert the linear wavelength transformation.

        Parameters
        ----------
        w : array_like
            Linear wavelength coordinates.

        Returns
        -------
        lam : ndarray
            Physical wavelengths.
        """
        return w + self.lam0

    def deriv(self, lam):
        """
        Compute dλ/dw for the linear wavelength transform.

        Parameters
        ----------
        lam : array_like
            Physical wavelengths.

        Returns
        -------
        dldw : float or ndarray
            Derivative of wavelength with respect to the transformed coordinate.
        """
        return 1.
    

class SNPITDisperser:
    """
    Spectral disperser model for the Roman/WFI, optimized to meet the needs
    of the SN PIT.

    This class loads a configuration file describing detector geometry
    and optical polynomial models, and provides methods to compute:

    - Dispersed positions on the Focal Plane Assembly (FPA),
    - Derivatives with respect to wavelength,
    - Local trace normals,
    - Local dispersion scales.

    All intermediate optical coordinates are expressed in Focal Plane
    Assembly (FPA) physical units and mapped to and from Sensor Chip
    Assembly (SCA) pixel coordinates.
    """

    def __init__(self, conffile):
        """
        Initialize the disperser from a configuration file.

        Parameters
        ----------
        conffile : str
            Path to the YAML configuration file.
        """
        self.load_conffile(conffile)

        # pre-compile the numba functions
        _, _ = self.disperse(1., 1., 1., 1, pairwise=True)
        _, _ = self.disperse(1., 1., 1., 1, pairwise=False)

        _, _ = self.deriv(1., 1., 1., 1, pairwise=True)
        _, _ = self.deriv(1., 1., 1., 1, pairwise=False)

        
    def load_conffile(self, conffile):
        """
        Load and parse the disperser configuration file.

        This method reads detector geometry, optical polynomial coefficients,
        and wavelength transformation settings, and precomputes derivative
        coefficients for efficient runtime evaluation.

        Parameters
        ----------
        conffile : str
            Path to the YAML configuration file.
        """
        self.conffile = conffile

        with open(self.conffile, 'r') as fp:
            cfg = yaml.safe_load(fp)

        # fetch the meta data
        self.meta = cfg['meta']

        # set the detector model
        self.detector = cfg['detector_model']

        # change some units
        self.detector['pixel_scale'] /= 3600.
        
        # set the optical model
        self.optical = {k: v for k, v in cfg['optical_model'].items()}

        # process each spectral order
        self.optical['orders'] = {}
        for order, data in cfg['optical_model']['orders'].items():
            
            # extract the parameters
            Xij = np.asarray(data['xmap_ij_coeff'])
            Yij = np.asarray(data['ymap_ij_coeff'])
            Cijk = np.asarray(data['ids_ijk_coeff'])
            Dijk = np.asarray(data['crv_ijk_coeff'])

            # check that the data are valid
            if Xij.all() and Yij.all() and Cijk.all() and Dijk.all():
                
                # compute derivatives
                dCijk = self.deriv_coeffs(Cijk)
                dDijk = self.deriv_coeffs(Dijk)

                # save the results
                self.optical['orders'][order] = {'Xij': Xij,
                                                'Yij': Yij,
                                                'Cijk': Cijk,
                                                'dCijk': dCijk,
                                                'Dijk': Dijk,
                                                'dDijk': dDijk}

        # transform the wavelengths            
        transform = self.optical['wl_transform'].lower()
        if transform == 'log':
            self.lam_transformer = LogTransformer(self.optical['wl_reference'])
        elif transform == 'linear':
            self.lam_transformer = LinearTransformer(self.optical['wl_reference'])
        else:
            raise NotImplementedError(f'Invalid transform {transform}')
            

    @staticmethod
    def deriv_coeffs(M):
        """
        Compute derivative polynomial coefficients with respect to the first
        dimension using Horner-compatible form.

        Parameters
        ----------
        M : ndarray, shape (n, ...)
            Polynomial coefficient tensor where the first axis corresponds
            to the variable of differentiation.

        Returns
        -------
        dM : ndarray, shape (n-1, ...)
            Derivative coefficient tensor.
        """
        n = M.shape[0]
        ii = np.arange(1, n, dtype=float)
        shape = (n-1,) + (1,) * (M.ndim - 1)
        dM = M[1:] * ii.reshape(shape)

        return dM

    def validate(self, x, y, l, sca, order='1', pairwise=True):
        """
        Validate input array dimensionality for disperser evaluation.

        Parameters
        ----------
        x, y : ndarray
            Undispersed FPA coordinates.
        l : ndarray
            Wavelength array.
        sca : int
            The SCA number.
        order : str
            The spectral order.
        pairwise : bool
            If True, require x, y, and l to have identical shapes.
            If False, require x and y to match and l to be one-dimensional.

        """
        if order not in self.optical['orders']:
            raise KeyError("Invalid spectral order")

        if sca not in self.detector['xy_centers']:
            raise KeyError("Invalid SCA number")

        if pairwise:
            if (x.shape != y.shape) or (x.shape != l.shape):
                raise RuntimeError("Invalid x, y, lam shape for pairwise")
        else:
            if (x.shape != y.shape) or (l.ndim != 1):
                raise RuntimeError("Invalid x, y, lam shape")            
                
    def sca_to_fpa(self, xsca, ysca, sca):
        """
        Convert Sensor Chip Assembly (SCA) pixel coordinates to
        Focal Plane Assembly (FPA) physical coordinates.

        Parameters
        ----------
        xsca, ysca : array_like
            Pixel coordinates on the SCA detector.
        sca : int or key
            Identifier for the detector segment.

        Returns
        -------
        xfpa, yfpa : ndarray
            Physical coordinates in the Focal Plane Assembly (FPA) frame.
        """
        dx = self.detector['crpix1']-xsca # WHY NEGATIVE?
        dy = ysca-self.detector['crpix2']

        xcen, ycen = self.detector['xy_centers'][sca]
         
        xfpa = (xcen*self.detector['plate_scale'] + dx)*self.detector['pixel_scale']
        yfpa = (ycen*self.detector['plate_scale'] + dy)*self.detector['pixel_scale']

        return xfpa, yfpa

    
    def mpa_to_sca(self, xmpa, ympa, sca):
        """
        Convert Mosaic Plate Assembly (MPA) physical coordinates back to
        Sensor Chip Assembly (SCA) pixel coordinates.

        Parameters
        ----------
        xmpa, ympa : array_like
            Physical Mosaic Plate Assembly coordinates.
        sca : int or key
            Identifier for the detector segment.

        Returns
        -------
        xsca, ysca : ndarray
            Detector pixel coordinates.
        """
        xcen, ycen = self.detector['xy_centers'][sca]
        
        xoff = (xmpa - xcen)*self.detector['plate_scale']
        yoff = (ympa - ycen)*self.detector['plate_scale']

        xsca = self.detector['crpix1'] - xoff # WHY NEGATIVE?
        ysca = self.detector['crpix2'] + yoff

        return xsca, ysca

    def disperse(self, x0, y0, lam, sca, order='1', pairwise=False):
        """
        Compute dispersed detector coordinates for given source positions
        and wavelengths.

        Parameters
        ----------
        x0, y0 : array_like
            Undispersed Sensor Chip Assembly (SCA) pixel coordinates.
        lam : array_like
            Physical wavelengths.
        sca : int or key
            Detector segment identifier.
        order : str, optional
            Spectral order identifier (default is '1').
        pairwise : bool, optional
            If True, treat x0, y0, and lam as pairwise-aligned arrays.
            If False, compute a full grid over x0/y0 and lam.

        Returns
        -------
        xp, yp : ndarray
            Dispersed SCA pixel coordinates.
        """
        x0 = np.atleast_1d(x0)
        y0 = np.atleast_1d(y0)
        lam = np.atleast_1d(lam)

        # check inputs
        self.validate(x0, y0, lam, sca, order=order, pairwise=pairwise)
        
        # convert SCA coordinates to FPA coordinates
        xfpa, yfpa = self.sca_to_fpa(x0, y0, sca)
                
        # transform wavelengths
        wtran = self.lam_transformer.evaluate(lam)
        
        if pairwise:
            xt, yt = numba_disperse_pairwise(
                self.optical['orders'][order]['Xij'],
                self.optical['orders'][order]['Yij'],
                self.optical['orders'][order]['Cijk'],
                self.optical['orders'][order]['Dijk'],
                xfpa, yfpa, wtran)
        else:
            xt, yt = numba_disperse(
                self.optical['orders'][order]['Xij'],
                self.optical['orders'][order]['Yij'],
                self.optical['orders'][order]['Cijk'],
                self.optical['orders'][order]['Dijk'],
                xfpa, yfpa, wtran)
            
            
        # convert back to SCA coordinates (this method should deal with
        # the traces that span multiple SCAs)
        xp, yp = self.mpa_to_sca(xt, yt, sca)
       
        return xp, yp

    def deriv(self, x0, y0, lam, sca, order='1', pairwise=False):
        """
        Compute derivatives of dispersed detector coordinates with respect
        to wavelength.

        Parameters
        ----------
        x0, y0 : array_like
            Undispersed Sensor Chip Assembly (SCA) pixel coordinates.
        lam : array_like
            Physical wavelengths.
        sca : int or key
            Detector segment identifier.
        order : str, optional
            Spectral order identifier (default is '1').
        pairwise : bool, optional
            If True, treat x0, y0, and lam as pairwise-aligned arrays.
            If False, compute a full grid over x0/y0 and lam.

        Returns
        -------
        dxdl, dydl : ndarray
            Derivatives of dispersed SCA pixel coordinates with respect to
            wavelength.
        """
        x0 = np.atleast_1d(x0)
        y0 = np.atleast_1d(y0)
        lam = np.atleast_1d(lam)

        # check inputs
        self.validate(x0, y0, lam, sca, order=order, pairwise=pairwise)
                
        # convert SCA coordinates to FPA coordinates
        xfpa, yfpa = self.sca_to_fpa(x0, y0, sca)
                
        # transform wavelengths
        wtran = self.lam_transformer.evaluate(lam)
        dldw = self.lam_transformer.deriv(lam)
        
        if pairwise:
            dxdw, dydw = numba_deriv_pairwise(
                self.optical['orders'][order]['dCijk'],
                self.optical['orders'][order]['Cijk'],
                self.optical['orders'][order]['dDijk'],
                xfpa, yfpa, wtran)

            dxdl = dxdw/dldw
            dydl = dydw/dldw
            
        else:
            dxdw, dydw = numba_deriv(
                self.optical['orders'][order]['dCijk'],
                self.optical['orders'][order]['Cijk'],
                self.optical['orders'][order]['dDijk'],
                xfpa, yfpa, wtran)

            dxdl = dxdw/dldw[np.newaxis, :]
            dydl = dydw/dldw[np.newaxis, :]

        dxdl *= self.detector['plate_scale']
        dydl *= self.detector['plate_scale']
        
        return dxdl, dydl

    def normal(self, *args, **kwargs):
        """
        Compute unit-normal vectors to the spectral trace.

        This returns vectors perpendicular to the dispersion direction
        at each sampled point.

        Returns
        -------
        nx, ny : ndarray
            Components of the normal vectors.
        """
        dxdl, dydl = self.deriv(*args, **kwargs)
        return dydl, -dxdl
            
    def dispersion(self, *args, **kwargs):
        """
        Compute the local dispersion scale (dλ/dr) along the spectral trace.

        Returns
        -------
        dldr : ndarray
            Local dispersion (wavelength per pixel).
        """
        dxdl, dydl = self.deriv(*args, **kwargs)
        dldr = 1./np.hypot(dxdl, dydl)
        return dldr

if __name__ == '__main__':

    # instantiate the disperser
    disp = SNPITDisperser('Roman_prism_OpticalModel_v0.8.yaml')

    # which SCA are we considering?
    sca = 1

    # length of test data
    N = 1000
    
    # create some random positions
    x = np.random.uniform(low=0, high=4088, size=N)
    y = np.random.uniform(low=0, high=4088, size=N)

    # assume one wavelength (in micron) for each position
    l = np.random.uniform(low=disp.optical['wl_min'],
        high=disp.optical['wl_max'], size=N)

    # pair-wise compute the dispersed positions:  here, every (x, y)
    # has exactly one corresponding wavelength (l), so that this maps the
    # tuples: (x,y,l) -> (xp, yp).  
    xp, yp = disp.disperse(x, y, l, sca, order='1', pairwise=True)
    
    # can also compute the derivatives, which is the tangent vector
    # at some point:
    dxdl, dydl = disp.deriv(x, y, l, sca, order='1', pairwise=True)

    # can compute the dispersion (the rate of change of the wavelength
    # on a path along the trace:
    dldr = disp.dispersion(x, y, l, sca, order='1', pairwise=True)
    
    # now demo the non-pairwise.  Here if (x,y) are vectors of size
    # N elements and l is a vector of M elements, now the output data
    # will be a 2d array of (N, M) elements        

    # create a new wavelength grid 
    l = np.linspace(disp.optical['wl_min'], disp.optical['wl_max'], 100)

    # new dispersed positions
    xp, yp = disp.disperse(x, y, l, sca, order='1', pairwise=False)

    # new tangent vectors
    dxdl, dydl = disp.deriv(x, y, l, sca, order='1', pairwise=False)

    # new dispersion
    dldr = disp.dispersion(x, y, l, sca, order='1', pairwise=False)

