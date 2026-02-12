import unittest
import numpy as np
import os
import tempfile
import yaml

# Import galsim and roman_imsim
import galsim
import galsim.config
from roman_imsim.photonOps import *


class TestPhotonOps(unittest.TestCase):
    """Unit tests for all registered photon operators."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Get the root directory of the package
        self.root_dir = os.path.dirname(os.path.dirname(__file__))
        
        # Create a mock photon array
        self.num_photons = 10
        self.mock_photon_array = self._create_mock_photon_array()
        
    def _create_mock_photon_array(self):
        photon_array = galsim.PhotonArray(self.num_photons)
        
        photon_array.x = np.random.uniform(-5, 5, self.num_photons)
        photon_array.y = np.random.uniform(-5, 5, self.num_photons)
        
        photon_array.flux = np.random.uniform(100, 1000, self.num_photons)
        
        photon_array.wavelength = np.random.uniform(500, 2000, self.num_photons)
        
        return photon_array
    

    def test_photon_operators_smoke(self):
        """Test that all registered roman_imsim photon operators can be initialized and applied."""
        photon_operators = [ChargeDiff, SlitlessSpec, GrismNV, GrismV, WFSSSDisperser]
        for photon_operator_cls in photon_operators:
            with self.subTest(operator=photon_operator_cls.__name__):
                photon_op = photon_operator_cls()
                photon_op.applyTo(self.mock_photon_array)



if __name__ == '__main__':
    unittest.main()
