import unittest
import numpy as np
import tempfile
import yaml
import os
import roman_imsim.photonOps._wfss_disperser.snpitdispenser


class TestSNPITDisperser(unittest.TestCase):
    """Unit tests for the SNPITDisperser class and related components."""
    
    def setUp(self):
        """Set up test fixtures with a minimal valid configuration."""
        
        # Initialize disperser
        root_dir = os.path.dirname(roman_imsim.__path__[0])
        self.config_file_name = os.path.join(root_dir, "optical_models/Roman_prism_OpticalModel_v0.8.yaml")
        self.disperser = roman_imsim.snpitdispenser.SNPITDisperser(os.path.join(root_dir, "optical_models/Roman_prism_OpticalModel_v0.8.yaml"))
        
    def test_load_conffile_invalid_transform(self):
        """Test that invalid wavelength transform raises error."""

        with open(self.config_file_name, 'r') as fp:
            config = yaml.safe_load(fp)
        config['optical_model']['wl_transform'] = 'invalid'
        
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        yaml.dump(config, temp_file)
        temp_file.close()
        
        with self.assertRaises(NotImplementedError):
            roman_imsim.snpitdispenser.SNPITDisperser(temp_file.name)
            
        os.unlink(temp_file.name)
        
    def test_deriv_coeffs(self):
        """Test polynomial coefficient derivative computation."""
        # Test 1D case
        M = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        expected = np.array([[4, 5, 6], [14, 16, 18]])  # [1*4, 2*5, 3*7], [2*7, 2*8, 2*9]
        result = roman_imsim.snpitdispenser.SNPITDisperser.deriv_coeffs(M)
        np.testing.assert_array_equal(result, expected)
        
        # Test 3D case
        M_3d = np.random.rand(4, 3, 2)
        result_3d = roman_imsim.snpitdispenser.SNPITDisperser.deriv_coeffs(M_3d)
        expected_shape = (3, 3, 2)
        self.assertEqual(result_3d.shape, expected_shape)
        
    def test_validate_inputs_valid(self):
        """Test input validation with valid inputs."""
        x = np.array([100, 200])
        y = np.array([150, 250])
        lam = np.array([1.0, 1.5])
        sca = 1
        
        # Should not raise any exceptions
        self.disperser.validate(x, y, lam, sca, order='1', pairwise=True)
        self.disperser.validate(x, y, lam, sca, order='1', pairwise=False)
        
    def test_validate_inputs_invalid_order(self):
        """Test input validation with invalid spectral order."""
        x = np.array([100, 180])
        y = np.array([120, 200])
        lam = np.array([1.0, 1.5])
        sca = 1
        
        with self.assertRaises(KeyError):
            self.disperser.validate(x, y, lam, sca, order='2', pairwise=True)
            
    def test_validate_inputs_invalid_sca(self):
        """Test input validation with invalid SCA number."""
        x = np.array([110, 200])
        y = np.array([100, 220])
        lam = np.array([1.0, 1.5])
        sca = 99  # Invalid SCA
        
        with self.assertRaises(KeyError):
            self.disperser.validate(x, y, lam, sca, order='1', pairwise=True)
            
    def test_validate_inputs_pairwise_shape_mismatch(self):
        """Test input validation with shape mismatch in pairwise mode."""
        x = np.array([105, 200])
        y = np.array([100, 210, 300])  # Different shape
        lam = np.array([1.0, 1.5])
        sca = 1
        
        with self.assertRaises(RuntimeError):
            self.disperser.validate(x, y, lam, sca, order='1', pairwise=True)
            
    def test_validate_inputs_non_pairwise_shape_mismatch(self):
        """Test input validation with shape mismatch in non-pairwise mode."""
        x = np.array([133, 245])
        y = np.array([101, 205, 303])  # Different shape
        lam = np.array([1.0, 1.2])  # Should be 1D
        sca = 1
        
        with self.assertRaises(RuntimeError):
            self.disperser.validate(x, y, lam, sca, order='1', pairwise=False)
            
    def test_sca_to_fpa_conversion(self):
        """Test SCA to FPA coordinate conversion."""
        xsca = np.array([2044.5])
        ysca = np.array([2044.5])
        sca = 1
        
        xfpa, yfpa = self.disperser.sca_to_fpa(xsca, ysca, sca)
        
        # Should return arrays
        self.assertIsInstance(xfpa, np.ndarray)
        self.assertIsInstance(yfpa, np.ndarray)
        
        # Should have same length as input
        self.assertEqual(len(xfpa), len(xsca))
        self.assertEqual(len(yfpa), len(ysca))
        
    def test_mpa_to_sca_conversion(self):
        """Test MPA to SCA coordinate conversion."""
        xmpa = np.array([0.0])
        ympa = np.array([0.0])
        sca = 1
        
        xsca, ysca = self.disperser.mpa_to_sca(xmpa, ympa, sca)
        
        # Should return arrays
        self.assertIsInstance(xsca, np.ndarray)
        self.assertIsInstance(ysca, np.ndarray)
        
        # Should have same length as input
        self.assertEqual(len(xsca), len(xmpa))
        self.assertEqual(len(ysca), len(ympa))
        
    @unittest.skip("Unclear if this should pass.")
    def test_coordinate_conversion_roundtrip(self):
        """Test that coordinate conversion is approximately reversible."""
        xsca_orig = np.array([1234.0, 2000.0])
        ysca_orig = np.array([1000.0, 2567.0])
        sca = 1
        
        # Convert SCA -> FPA -> SCA
        xfpa, yfpa = self.disperser.sca_to_fpa(xsca_orig, ysca_orig, sca)
        xsca_back, ysca_back = self.disperser.mpa_to_sca(xfpa, yfpa, sca)
        
        # Should be approximately equal (allowing for numerical precision)
        np.testing.assert_allclose(xsca_orig, xsca_back, rtol=1e-10)
        np.testing.assert_allclose(ysca_orig, ysca_back, rtol=1e-10)
        
    def test_disperse_pairwise(self):
        """Test dispersion computation in pairwise mode."""
        x = np.array([1020.0, 2048.0])
        y = np.array([2040.0, 1000.0])
        lam = np.array([1.0, 1.5])
        sca = 1
        
        xp, yp = self.disperser.disperse(x, y, lam, sca, order='1', pairwise=True)
        
        # Should return arrays
        self.assertIsInstance(xp, np.ndarray)
        self.assertIsInstance(yp, np.ndarray)
        
        # Should have same shape as input
        self.assertEqual(xp.shape, x.shape)
        self.assertEqual(yp.shape, y.shape)
        
        # Should be different from input (dispersion should change positions)
        self.assertFalse(np.allclose(xp, x))
        self.assertFalse(np.allclose(yp, y))
        
    def test_disperse_non_pairwise(self):
        """Test dispersion computation in non-pairwise mode."""
        x = np.array([1000.0, 2000.0])
        y = np.array([1000.0, 2000.0])
        lam = np.array([1.0, 1.5])  # 1D wavelength array
        sca = 1
        
        xp, yp = self.disperser.disperse(x, y, lam, sca, order='1', pairwise=False)
        
        # Should return 2D arrays
        self.assertEqual(xp.ndim, 2)
        self.assertEqual(yp.ndim, 2)
        
        # Should have shape (len(x), len(lam))
        expected_shape = (len(x), len(lam))
        self.assertEqual(xp.shape, expected_shape)
        self.assertEqual(yp.shape, expected_shape)
        
    def test_deriv_pairwise(self):
        """Test derivative computation in pairwise mode."""
        x = np.array([1000.0, 2000.0])
        y = np.array([1000.0, 2000.0])
        lam = np.array([1.0, 1.5])
        sca = 1
        
        dxdl, dydl = self.disperser.deriv(x, y, lam, sca, order='1', pairwise=True)
        
        # Should return arrays
        self.assertIsInstance(dxdl, np.ndarray)
        self.assertIsInstance(dydl, np.ndarray)
        
        # Should have same shape as input
        self.assertEqual(dxdl.shape, x.shape)
        self.assertEqual(dydl.shape, y.shape)
        
        # Should be non-zero (dispersion should have non-zero derivative)
        self.assertFalse(np.allclose(dxdl, 0))
        self.assertFalse(np.allclose(dydl, 0))
        
    def test_deriv_non_pairwise(self):
        """Test derivative computation in non-pairwise mode."""
        x = np.array([1000.0, 2000.0])
        y = np.array([1000.0, 2000.0])
        lam = np.array([1.0, 1.5])  # 1D wavelength array
        sca = 1
        
        dxdl, dydl = self.disperser.deriv(x, y, lam, sca, order='1', pairwise=False)
        
        # Should return 2D arrays
        self.assertEqual(dxdl.ndim, 2)
        self.assertEqual(dydl.ndim, 2)
        
        # Should have shape (len(x), len(lam))
        expected_shape = (len(x), len(lam))
        self.assertEqual(dxdl.shape, expected_shape)
        self.assertEqual(dydl.shape, expected_shape)
        
    def test_normal(self):
        """Test normal vector computation."""
        x = np.array([1000.0, 2000.0])
        y = np.array([1000.0, 2000.0])
        lam = np.array([1.0, 1.5])
        sca = 1
        
        nx, ny = self.disperser.normal(x, y, lam, sca, order='1', pairwise=True)
        
        # Should return arrays
        self.assertIsInstance(nx, np.ndarray)
        self.assertIsInstance(ny, np.ndarray)
        
        # Should have same shape as input
        self.assertEqual(nx.shape, x.shape)
        self.assertEqual(ny.shape, y.shape)
        
        # Normal vectors should be perpendicular to derivative vectors
        dxdl, dydl = self.disperser.deriv(x, y, lam, sca, order='1', pairwise=True)
        dot_product = dxdl * nx + dydl * ny
        np.testing.assert_allclose(dot_product, 0, atol=1e-10)
        
    def test_dispersion(self):
        """Test dispersion scale computation."""
        x = np.array([1000.0, 2000.0])
        y = np.array([1000.0, 2000.0])
        lam = np.array([1.0, 1.5])
        sca = 1
        
        dldr = self.disperser.dispersion(x, y, lam, sca, order='1', pairwise=True)
        
        # Should return array
        self.assertIsInstance(dldr, np.ndarray)
        
        # Should have same shape as input
        self.assertEqual(dldr.shape, x.shape)
        
        # Should be positive (dispersion scale should be positive)
        self.assertTrue(np.all(dldr > 0))
        
    def test_scalar_inputs(self):
        """Test that scalar inputs are handled correctly."""
        x = 1000.0
        y = 1000.0
        lam = 1.0
        sca = 1
        
        # Should work with scalar inputs
        xp, yp = self.disperser.disperse(x, y, lam, sca, order='1', pairwise=True)
        dxdl, dydl = self.disperser.deriv(x, y, lam, sca, order='1', pairwise=True)
        
        # Should return 1D arrays
        self.assertEqual(xp.ndim, 1)
        self.assertEqual(yp.ndim, 1)
        self.assertEqual(dxdl.ndim, 1)
        self.assertEqual(dydl.ndim, 1)
        
        # Should have length 1
        self.assertEqual(len(xp), 1)
        self.assertEqual(len(yp), 1)
        self.assertEqual(len(dxdl), 1)
        self.assertEqual(len(dydl), 1)


class TestLogTransformer(unittest.TestCase):
    """Unit tests for the LogTransformer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.lam0 = 1.0
        self.transformer = roman_imsim.snpitdispenser.LogTransformer(self.lam0)
        
    def test_initialization(self):
        """Test transformer initialization."""
        self.assertEqual(self.transformer.lam0, self.lam0)
        self.assertAlmostEqual(self.transformer.ln10, np.log(10.0))
        
    def test_evaluate(self):
        """Test wavelength transformation."""
        lam = np.array([0.5, 1.0, 2.0])
        w = self.transformer.evaluate(lam)
        
        expected = np.log10(lam / self.lam0)
        np.testing.assert_allclose(w, expected)
        
    def test_invert(self):
        """Test inverse transformation."""
        w = np.array([-0.3, 0.0, 0.3])
        lam = self.transformer.invert(w)
        
        expected = self.lam0 * (10.0 ** w)
        np.testing.assert_allclose(lam, expected)
        
    def test_roundtrip(self):
        """Test that evaluate and invert are inverses."""
        lam_orig = np.array([0.5, 1.0, 2.0])
        
        w = self.transformer.evaluate(lam_orig)
        lam_back = self.transformer.invert(w)
        
        np.testing.assert_allclose(lam_orig, lam_back, rtol=1e-10)
        
    def test_deriv(self):
        """Test derivative computation."""
        lam = np.array([0.5, 1.0, 2.0])
        dldw = self.transformer.deriv(lam)
        
        expected = lam * np.log(10.0)
        np.testing.assert_allclose(dldw, expected)


class TestLinearTransformer(unittest.TestCase):
    """Unit tests for the LinearTransformer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.lam0 = 1.0
        self.transformer = roman_imsim.snpitdispenser.LinearTransformer(self.lam0)
        
    def test_initialization(self):
        """Test transformer initialization."""
        self.assertEqual(self.transformer.lam0, self.lam0)
        
    def test_evaluate(self):
        """Test wavelength transformation."""
        lam = np.array([0.5, 1.0, 2.0])
        w = self.transformer.evaluate(lam)
        
        expected = lam - self.lam0
        np.testing.assert_allclose(w, expected)
        
    def test_invert(self):
        """Test inverse transformation."""
        w = np.array([-0.5, 0.0, 1.0])
        lam = self.transformer.invert(w)
        
        expected = w + self.lam0
        np.testing.assert_allclose(lam, expected)
        
    def test_roundtrip(self):
        """Test that evaluate and invert are inverses."""
        lam_orig = np.array([0.5, 1.0, 2.0])
        
        w = self.transformer.evaluate(lam_orig)
        lam_back = self.transformer.invert(w)
        
        np.testing.assert_allclose(lam_orig, lam_back, rtol=1e-10)
        
    def test_deriv(self):
        """Test derivative computation."""
        lam = np.array([0.5, 1.0, 2.0])
        dldw = self.transformer.deriv(lam)
        
        # Derivative should always be 1 for linear transform
        expected = 1.0
        np.testing.assert_allclose(dldw, expected)


if __name__ == '__main__':
    unittest.main()