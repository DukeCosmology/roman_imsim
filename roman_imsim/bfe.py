import fitsio as fio
import numpy as np
from galsim import PhotonArray

# This will eventually be a class in PhotonOps
# but for now this is a standalone script to test the BFE effect


#
#---------------------------------------------------------------------------------------------
# Brighter-Fatter Effect (BFE) implementation

# This code is partially adapted from Chien-Hao's example in the Roman ImSim repo where he wrote the BFE effect
# applied as a photonOps class but here we are doing it as a photon by photon operation
# find the original code here as of 8/12/25: roman_imsim/Effects/BrighterFatter.py (branch: inc_det_into_sca)

#         Apply brighter-fatter effect.
#         Brighter fatter effect is a non-linear effect that deflects photons due to the
#         the eletric field built by the accumulated charges. This effect exists in both
#         CCD and CMOS detectors and typically percent level change in charge.
#         The built-in electric field by the charges in pixels tends to repulse charges
#         to nearby pixels. Thus, the profile of more illuminous ojbect becomes broader.
#         This effect can also be understood effectly as change in pixel area and pixel
#         boundaries.
#         BFE is defined in terms of the Antilogus coefficient kernel of total pixel area change
#         in the detector effect charaterization file. Kernel of the total pixel area, however,
#         is not sufficient. Image simulation of the brighter fatter effect requires the shift
#         of the four pixel boundaries. Before we get better data, we solve for the boundary
#         shift components from the kernel of total pixel area by assumming several symmetric constraints.


# The current plan is for the order to go like this:
#      1. Take in the pre-readout image, which has the accumulated charge from all photons 
#         (BFE will have been applied to the first pre-readout image using some probability distribution)
#      2. Take in new photon array that is to be added to the pre-readout image
#      3. For each photon array, use the pre-readout image to determine the charge distribution
#         and then apply the BFE effect to that photon array
#      4. After all photons have been processed, then do the readout step
#         to stack the previous pre-readout image and the new photons
#      5. Repeat, Output the final image

