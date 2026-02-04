import yaml
import numpy as np

import numba as nb


@nb.njit
def horner1_scalar(c, x):
    ''' implement 1d horner method assuming x is a scalar '''
    n = len(c)
    y = c[-1]
    for i in range(n-2,-1, -1):
        y = y*x+c[i]
    return y

@nb.njit
def horner2_scalar(c, x, y):
    ''' implement 2d horner method assuming x,y are scalars '''
    n, m = c.shape
    z = horner1_scalar(c[:,-1], x)
    for j in range(m-2, -1, -1):
        z = z*y + horner1_scalar(c[:, j], x)
    return z

@nb.njit
def horner1_vector(c, x):
    ''' implement 1d horner method assuming x is a np.ndarray '''
    n = len(c)
    y = np.full_like(x, c[-1])
    for i in range(n-2,-1, -1):
        y = y*x+c[i]
    return y

@nb.njit
def horner2_vector(c, x, y):
    ''' implement 2d horner method assuming x, y are np.ndarray '''
    n, m = c.shape
    z = horner1_vector(c[:,-1], x)
    for j in range(m-2, -1, -1):
        z = z*y + horner1_vector(c[:, j], x)
    return z

@nb.njit
def horner3(c, x, y, dl):
    ''' implement 3d horner method assuming x, y are scalar but dl is
    a np.ndarray '''

    nl = len(dl)
    n = c.shape[0]
    
    f = np.full(nl, horner2_scalar(c[-1, :, :], x, y))
    for i in range(n-2, -1, -1):
        f = f*dl + horner2_scalar(c[i, :, :], x, y)

    return f
    

@nb.njit(parallel=False)
def relative_trace(Cijk, Dijk, xtrc, ytrc, wtran):
    ''' compute the position along the trace.

    I believe this is equivalent to xarr and yarr in VM's code
    '''
    
    npix = len(xtrc)
    nlam = len(wtran)

    dx = np.empty((npix, nlam), dtype=float)
    dy = np.empty((npix, nlam), dtype=float)
    
    for i in range(npix):
        xt = xtrc[i]
        yt = ytrc[i]

        # Vihang calls these "yarr" and "xarr"
        dely = horner3(Cijk, xt, yt, wtran)
        delx = horner3(Dijk, xt, yt, dely)
        
        # save results
        dy[i] = dely
        dx[i] = delx

    return dx, dy

@nb.njit(parallel=False)
def dispersion(dDijk, xtrc, ytrc, dl):
    ''' compute the dispersion (ie. dlambda/dy) '''
    
    nd = dDijk.shape[0]
    
    npix = len(xtrc)
    nlam = len(dl)

    dydl = np.empty((npix, nlam), dtype=float)
    
    for i in nb.prange(npix):
        xt = xtrc[i]
        yt = ytrc[i]
        dely = np.full(nlam, horner2_scalar(dDijk[-1, :, :], xt, yt))
        for ii in range(nd-2, -1, -1):
            dely = dely*dl + horner2_scalar(dDijk[ii, :, :], xt, yt)

        dydl[i] = dely
    return dydl


class SNPITDisperser:
    def __init__(self, conffile):
        self.load_conffile(conffile)

    def load_conffile(self, conffile):
        self.conffile = conffile

        with open(self.conffile, 'r') as fp:
            cfg = yaml.load(fp, Loader=yaml.FullLoader)

        self.meta = cfg['meta']
        self.detector = cfg['detector_model']
        self.optical = {}
        self.optical['wl_min'] = cfg['optical_model']['wl_min']
        self.optical['wl_max'] = cfg['optical_model']['wl_max']
        self.optical['wl_reference'] = cfg['optical_model']['wl_reference']
        self.optical['wl_transform'] = cfg['optical_model']['wl_transform']
        self.optical['orders'] = {}
        for order, data in cfg['optical_model']['orders'].items():
            self.optical['orders'][order] = {'xij': np.asarray(data['xmap_ij_coeff']),
                                            'yij': np.asarray(data['ymap_ij_coeff']),
                                            'crv': np.asarray(data['crv_ijk_coeff']),
                                            'ids': np.asarray(data['ids_ijk_coeff'])}
            
                
    def sca_to_fpa(self, xsca, ysca, sca):
        dx = self.detector['crpix1']-xsca # WHY NEGATIVE?!?!?
        dy = ysca-self.detector['crpix2']

        xcen, ycen = self.detector['xy_centers'][sca]
 
        
        xfpa = (xcen*self.detector['plate_scale'] + dx)*self.detector['pixel_scale']/3600.

        yfpa = (ycen*self.detector['plate_scale'] + dy)*self.detector['pixel_scale']/3600.

        return xfpa, yfpa

    def fpa_to_mpa(self, xfpa, yfpa, order):

        xmpa = horner2_vector(self.optical['orders'][order]['xij'], xfpa, yfpa)
        ympa = horner2_vector(self.optical['orders'][order]['yij'], xfpa, yfpa)
        
        return xmpa, ympa

    def mpa_to_sca(self, xmpa, ympa, sca):
        '''
        this function will need to be generalized to deal with traces that
        span multiple SCAs
        '''
        
        xcen, ycen = self.detector['xy_centers'][sca]
        
        xoff = (xmpa - xcen)*self.detector['plate_scale']
        yoff = (ympa - ycen)*self.detector['plate_scale']

        xsca = self.detector['crpix1'] - xoff
        ysca = self.detector['crpix2'] + yoff

        return xsca, ysca



    def transform_wavelength(self, ww, inverse=False):
        transform = self.optical['wl_transform']
        reference = self.optical['wl_reference']
        if transform == 'log':
            if inverse:
                return reference*(10.**ww)
            else:
                return np.log10(ww/reference)
            
        elif transform == 'linear':
            if inverse:
                return ww+reference
            else:
                return ww-reference
        else:
            raise NotImplementedError(f"Invalid transformation {transform}")
        
    def disperse(self, x0, y0, lam, sca, order='1'):

        '''
        inputs:
          x0, y0: coordinates in the SCA in pixels.
          lam: wavelength in microns
          sca: the SCA numer
        outputs:
          x, y: dispersed coordinates in the SCA in pixels

        issues:
          not sure how this will work for traces that span two SCAs
        
        '''
        x0 = np.atleast_1d(x0)
        y0 = np.atleast_1d(y0)
        lam = np.atleast_1d(lam)

        # convert SCA coordinates to FPA coordinates
        xfpa, yfpa = self.sca_to_fpa(x0, y0, sca)
                
        # transform wavelengths
        wtran = self.transform_wavelength(lam)
            
        # compute the relative trace
        dx, dy = relative_trace(self.optical['orders'][order]['ids'],
            self.optical['orders'][order]['crv'], xfpa, yfpa, wtran)

        # apply trace.  But need the MPA coordinates
        xmpa, ympa = self.fpa_to_mpa(xfpa, yfpa, order)
        
        # apply trace in the MPA system
        xt = xmpa[:, np.newaxis] + dx
        yt = ympa[:, np.newaxis] + dy

        # convert back to SCA coordinates (this method should deal with
        # the traces that span multiple SCAs)
        xp, yp = self.mpa_to_sca(xt, yt, sca)
       
        return xp, yp