import os
import numpy as np
import fitsio as fio
from roman_imsim.effects import roman_effects
from .utils import sca_number_to_file

class brighter_fatter(roman_effects):
    def __init__(self, params, base, logger, rng, rng_iter=None):
        super().__init__(params, base, logger, rng, rng_iter)
        # self.saturation_level = self.params['saturation_level'] if 'saturation_level' in self.params else 100000
        
        self.model = getattr(self, self.params['model'])
        if self.model is None:
            self.logger.warning("%s hasn't been implemented yet, the simple model will be applied for %s"%(str(self.params['model']), str(self.__class__.__name__)))
            self.model = self.simple_model
        
    def simple_model(self, image):
        self.logger.info("No bfe effect will be applied.")
        return image
    
    def lab_model(self, image):
        """
        Apply brighter-fatter effect.
        Brighter fatter effect is a non-linear effect that deflects photons due to the
        the eletric field built by the accumulated charges. This effect exists in both
        CCD and CMOS detectors and typically percent level change in charge.
        The built-in electric field by the charges in pixels tends to repulse charges
        to nearby pixels. Thus, the profile of more illuminous ojbect becomes broader.
        This effect can also be understood effectly as change in pixel area and pixel
        boundaries.
        BFE is defined in terms of the Antilogus coefficient kernel of total pixel area change
        in the detector effect charaterization file. Kernel of the total pixel area, however,
        is not sufficient. Image simulation of the brighter fatter effect requires the shift
        of the four pixel boundaries. Before we get better data, we solve for the boundary
        shift components from the kernel of total pixel area by assumming several symmetric constraints.
        Input
        im                                      : Image
        BFE[nbfe+Delta y, nbfe+Delta x, y, x]   : bfe coefficient kernel, nbfe=2
        """
        if self.sca_filepath is None:
            self.logger.warning("No BFE kernel data file provided; no bfe effect will be applied.")
            return image
        
        self.logger.warning("Lab measured model will be applied for brighter-fatter effect.")

        self.df = fio.FITS(os.path.join(self.sca_filepath, sca_number_to_file[self.sca]))

        nbfe = 2 ## kernel of bfe in shape (2 x nbfe+1)*(2 x nbfe+1)
        bin_size = 128
        n_max = 32
        m_max = 32
        num_grids = 4
        n_sub = n_max//num_grids
        m_sub = m_max//num_grids

        ##=======================================================================
        ##     solve boundary shfit kernel aX components
        ##=======================================================================
        a_area = self.df['BFE'][:,:,:,:] #5x5x32x32
        a_components = np.zeros( (4, 2*nbfe+1, 2*nbfe+1, n_max, m_max) ) #4x5x5x32x32

        ##solve aR aT aL aB for each a
        for n in range(n_max): #m_max and n_max = 32 (binned in 128x128)
            for m in range(m_max):
                a = a_area[:,:, n, m] ## a in (2 x nbfe+1)*(2 x nbfe+1)

                ## assume two parity symmetries
                a = ( a + np.fliplr(a) + np.flipud(a) + np.flip(a)  )/4.

                r = 0.5* ( 3.25/4.25  )**(1.5) / 1.5   ## source-boundary projection
                B = (a[2,2], a[3,2], a[2,3], a[3,3],
                    a[4,2], a[2,4], a[3,4], a[4,4] )

                A = np.array( [ [ -2 , -2 ,  0 ,  0 ,  0 ,  0 ,  0 ],
                                [  0 ,  1 ,  0 , -1 , -2 ,  0 ,  0 ],
                                [  1 ,  0 , -1 ,  0 , -2 ,  0 ,  0 ],
                                [  0 ,  0 ,  0 ,  0 ,  2 , -2 ,  0 ],
                                [  0 ,  0 ,  0 ,  1 ,  0 ,-2*r,  0 ],
                                [  0 ,  0 ,  1 ,  0 ,  0 ,-2*r,  0 ],
                                [  0 ,  0 ,  0 ,  0 ,  0 , 1+r, -1 ],
                                [  0 ,  0 ,  0 ,  0 ,  0 ,  0 ,  2 ]  ])


                s1,s2,s3,s4,s5,s6,s7 = np.linalg.lstsq(A, B, rcond=None)[0]

                aR = np.array( [[ 0.   , -s7  ,-r*s6 , r*s6 ,  s7  ],
                                [ 0.   , -s6  , -s5  ,  s5  ,  s6  ],
                                [ 0.   , -s3  , -s1  ,  s1  ,  s3  ],
                                [ 0.   , -s6  , -s5  ,  s5  ,  s6  ],
                                [ 0.   , -s7  ,-r*s6 , r*s6 ,  s7  ],])


                aT = np.array( [[   0.  ,  0. ,  0.  ,   0. ,   0.   ],
                                [  -s7  , -s6 , -s4  , -s6  ,  -s7   ],
                                [ -r*s6 , -s5 , -s2  , -s5  , -r*s6  ],
                                [  r*s6 ,  s5 ,  s2  ,  s5  ,  r*s6  ],
                                [   s7  ,  s6 ,  s4  ,  s6  ,   s7   ],])


                aL = aR[::-1, ::-1]
                aB = aT[::-1, ::-1]




                a_components[0, :,:, n, m] = aR[:,:]
                a_components[1, :,:, n, m] = aT[:,:]
                a_components[2, :,:, n, m] = aL[:,:]
                a_components[3, :,:, n, m] = aB[:,:]

        ##=============================
        ## Apply bfe to image
        ##=============================

        ## pad and expand kernels
        ## The img is clipped by the saturation level here to cap the brighter fatter effect and avoid unphysical behavior

        # array_pad = image.copy().array
        # saturation_array = np.ones_like(array_pad) * self.saturation_level
        # where_sat = np.where(array_pad > saturation_array)
        # array_pad[ where_sat ] = saturation_array[ where_sat ]
        # array_pad = array_pad[4:-4,4:-4]
        saturate = self.cross_refer('saturate')
        array_pad = saturate.apply(image = image.copy()).array[4:-4,4:-4] # img of interest 4088x4088
        array_pad = np.pad(array_pad, [(4+nbfe,4+nbfe),(4+nbfe,4+nbfe)], mode='symmetric') #4100x4100 array


        dQ_components = np.zeros( (4, bin_size*n_max, bin_size*m_max) )   #(4, 4096, 4096) in order of [aR, aT, aL, aB]


        ### run in sub grids to reduce memory

        ## pad and expand kernels
        t = np.zeros((bin_size*n_sub, n_sub))
        for row in range(t.shape[0]):
            t[row, row//(bin_size) ] =1



        for gj in range(num_grids):
            for gi in range(num_grids):

                a_components_pad = np.zeros( (4, 2*nbfe+1, 2*nbfe+1, bin_size*n_sub+2*nbfe, bin_size*m_sub+2*nbfe)  ) #(4,5,5,sub_grid,sub_grid)


                for comp in range(4):
                    for j in range(2*nbfe+1):
                        for i in range(2*nbfe+1):
                            tmp = (t.dot(  a_components[comp,j,i,gj*n_sub:(gj+1)*n_sub,gi*m_sub:(gi+1)*m_sub]  ) ).dot(t.T) #sub_grid*sub_grid
                            a_components_pad[comp, j, i, :, :] = np.pad(tmp, [(nbfe,nbfe),(nbfe,nbfe)], mode='symmetric')

                #convolve aX_ij with Q_ij
                for comp in range(4):
                    for dy in range(-nbfe, nbfe+1):
                        for dx in range(-nbfe, nbfe+1):
                            dQ_components[comp, gj*bin_size*n_sub : (gj+1)*bin_size*n_sub , gi*bin_size*m_sub : (gi+1)*bin_size*m_sub]\
                        += a_components_pad[comp, nbfe+dy, nbfe+dx,  nbfe-dy:nbfe-dy+bin_size*n_sub, nbfe-dx:nbfe-dx+bin_size*m_sub ]\
                            *array_pad[  -dy + nbfe + gj*bin_size*n_sub :  -dy + nbfe+ (gj+1)*bin_size*n_sub  ,  -dx + nbfe + gi*bin_size*m_sub : -dx + nbfe + (gi+1)*bin_size*m_sub ]

                    dj = int(np.sin(comp*np.pi/2))
                    di = int(np.cos(comp*np.pi/2))

                    dQ_components[comp, gj*bin_size*n_sub : (gj+1)*bin_size*n_sub , gi*bin_size*m_sub : (gi+1)*bin_size*m_sub]\
                    *= 0.5*(array_pad[   nbfe + gj*bin_size*n_sub :    nbfe+ (gj+1)*bin_size*n_sub  ,    nbfe + gi*bin_size*m_sub :    nbfe + (gi+1)*bin_size*m_sub ] +\
                            array_pad[dj+nbfe + gj*bin_size*n_sub : dj+nbfe+ (gj+1)*bin_size*n_sub  , di+nbfe + gi*bin_size*m_sub : di+nbfe + (gi+1)*bin_size*m_sub]  )

        image.array[:,:]  -= dQ_components.sum(axis=0)
        image.array[:,1:] += dQ_components[0][:,:-1]
        image.array[1:,:] += dQ_components[1][:-1,:]
        image.array[:,:-1] += dQ_components[2][:,1:]
        image.array[:-1,:] += dQ_components[3][1:,:]

        return image