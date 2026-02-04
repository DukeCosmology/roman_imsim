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
def relative_trace(Cijk, Dijk, x0, y0, wtran):
    ''' compute the position along the trace.

    I believe this is equivalent to xarr and yarr in VM's code
    '''
    
    npix = len(x0)
    nlam = len(wtran)

    dx = np.empty((npix, nlam), dtype=float)
    dy = np.empty((npix, nlam), dtype=float)
    
    for i in range(npix):
        xx0 = x0[i]
        yy0 = y0[i]

        # Vihang calls these "yarr" and "xarr"
        dely = horner3(Cijk, xx0, yy0, wtran)
        delx = horner3(Dijk, xx0, yy0, dely)
        
        # save results
        dy[i] = dely
        dx[i] = delx

    return dx, dy

@nb.njit
def dispersion(dCijk, Cijk, dDijk, x0, y0, wtran):
    npix = len(x0)
    nlam = len(wtran)

    dwdr = np.empty((npix, nlam), dtype=float)

    for i in range(npix):
        xx0 = x0[i]
        yy0 = y0[i]

        # compute the position
        dely = horner3(Cijk, xx0, yy0, wtran)
        
        # compute the derivatives        
        dydl = horner3(dCijk, xx0, yy0, wtran)
        dxdy = horner3(dDijk, xx0, yy0, dely)
        
        # use chain rule
        dwdr[i] = 1/(dydl*np.sqrt(dxdy**2+1))
        
    return dwdr


class LogTransformer:
    ln10 = 2.302585092994046

    def __init__(self, lam0):
        self.lam0 = lam0

    def evaluate(self, lam):
        return np.log10(lam/self.lam0)

    def invert(self, w):
        return self.lam0 * 10**w

    def deriv(self, lam):
        return lam*self.ln10
    
class LinearTransformer:
    
    def __init__(self, lam0):
        self.lam0 = lam0

    def evaluate(self, lam):
        return lam-self.lam0

    def invert(self, w):
        return lam+self.lam0

    def deriv(self, lam):
        return 1.
    

class SNPITDisperser:
    def __init__(self, conffile):
        self.load_conffile(conffile)

    def load_conffile(self, conffile):
        self.conffile = conffile

        with open(self.conffile, 'r') as fp:
            cfg = yaml.load(fp, Loader=yaml.FullLoader)

        # fetch the meta data
        self.meta = cfg['meta']

        # set the detector model
        self.detector = cfg['detector_model']

        # change some units
        self.detector['pixel_scale'] /= 3600.
        
        # set the optical model
        self.optical = {}
        self.optical['wl_min'] = cfg['optical_model']['wl_min']
        self.optical['wl_max'] = cfg['optical_model']['wl_max']
        self.optical['wl_reference'] = cfg['optical_model']['wl_reference']
        self.optical['wl_transform'] = cfg['optical_model']['wl_transform']
        self.optical['orders'] = {}
        for order, data in cfg['optical_model']['orders'].items():
            
            
            Xij = np.asarray(data['xmap_ij_coeff'])
            Yij = np.asarray(data['ymap_ij_coeff'])
            Cijk = np.asarray(data['ids_ijk_coeff'])
            Dijk = np.asarray(data['crv_ijk_coeff'])

            if Xij.all() and Yij.all() and Cijk.all() and Dijk.all():
                
                # save the results
                self.optical['orders'][order] = {'Xij': Xij,
                                                'Yij': Yij,
                                                'Cijk': Cijk,
                                                'dCijk': self.deriv_coeffs(Cijk),
                                                'Dijk': Dijk,
                                                'dDijk': self.deriv_coeffs(Dijk)}

            
        transform = self.optical['wl_transform'].lower()
        if transform == 'log':
            self.lam_transformer = LogTransformer(self.optical['wl_reference'])
        elif transform == 'linear':
            self.lam_transformer = LinearTransformer(self.optical['wl_reference'])
        else:
            raise NotImplementedError(f'Invalid transform {transform}')
            
    @staticmethod
    def deriv_coeffs(M):
        n = M.shape[0]
        ii = np.arange(1, n, dtype=float)
        dM = M[1:, :, :]*ii[:, np.newaxis, np.newaxis]
        return dM
                
    def sca_to_fpa(self, xsca, ysca, sca):
        dx = self.detector['crpix1']-xsca # WHY NEGATIVE?
        dy = ysca-self.detector['crpix2']

        xcen, ycen = self.detector['xy_centers'][sca]
         
        xfpa = (xcen*self.detector['plate_scale'] + dx)*self.detector['pixel_scale']
        yfpa = (ycen*self.detector['plate_scale'] + dy)*self.detector['pixel_scale']

        return xfpa, yfpa

    def fpa_to_mpa(self, xfpa, yfpa, order):

        xmpa = horner2_vector(self.optical['orders'][order]['Xij'], xfpa, yfpa)
        ympa = horner2_vector(self.optical['orders'][order]['Yij'], xfpa, yfpa)
        
        return xmpa, ympa

    def mpa_to_sca(self, xmpa, ympa, sca):
        '''
        this function will need to be generalized to deal with traces that
        span multiple SCAs
        '''
        
        xcen, ycen = self.detector['xy_centers'][sca]
        
        xoff = (xmpa - xcen)*self.detector['plate_scale']
        yoff = (ympa - ycen)*self.detector['plate_scale']

        xsca = self.detector['crpix1'] - xoff # WHY NEGATIVE?
        ysca = self.detector['crpix2'] + yoff

        return xsca, ysca


    def dispersion(self, x0, y0, lam, sca, order='1'):
        # coordinates will be 1d with N terms
        x0 = np.atleast_1d(x0)
        y0 = np.atleast_1d(y0)
        lam = np.atleast_1d(lam)

        # convert SCA coordinates to FPA coordinates
        xfpa, yfpa = self.sca_to_fpa(x0, y0, sca)

        # transform wavelengths
        wtran = self.lam_transformer.evaluate(lam)
        
        # compute the derivative
        dwdt = dispersion(self.optical['orders'][order]['dCijk'],
            self.optical['orders'][order]['Cijk'],
            self.optical['orders'][order]['dDijk'],
            xfpa, yfpa, wtran)
        
        # the derivative is dwtrans/dtheta.  so gotta transform
        # wtrans back to lambda and theta to pixel, so compute those Jacobians.
        dldw = self.lam_transformer.deriv(lam)
        drdt = self.detector['plate_scale']

        # we only want the positive side of things
        dldr = np.abs(dwdt)*dldw/drdt
        
        return dldr

        
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
        wtran = self.lam_transformer.evaluate(lam)
        
        #wtran = self.transform_wavelength(lam)
            
        # compute the relative trace
        dx, dy = relative_trace(self.optical['orders'][order]['Cijk'],
            self.optical['orders'][order]['Dijk'], xfpa, yfpa, wtran)
            
        # apply trace.  But need the MPA coordinates
        xmpa, ympa = self.fpa_to_mpa(xfpa, yfpa, order)
        
        # apply trace in the MPA system
        xt = xmpa[:, np.newaxis] + dx
        yt = ympa[:, np.newaxis] + dy

        # convert back to SCA coordinates (this method should deal with
        # the traces that span multiple SCAs)
        xp, yp = self.mpa_to_sca(xt, yt, sca)
       
        return xp, yp