import unittest
import numpy as np
from faulkner.activation import sigmoid

class TestSigmoid(unittest.TestCase):
    
    def test_sigmoid_positive_values(self):
        """Test sigmoid with positive values"""
        input_array = np.array([1, 2, 3])
        expected_output = 1 / (1 + np.exp(-input_array))
        np.testing.assert_array_almost_equal(sigmoid(input_array), expected_output, decimal=6)
    
    def test_sigmoid_negative_values(self):
        """Test sigmoid with negative values"""
        input_array = np.array([-1, -2, -3])
        expected_output = 1 / (1 + np.exp(-input_array))
        np.testing.assert_array_almost_equal(sigmoid(input_array), expected_output, decimal=6)
    
    def test_sigmoid_zero(self):
        """Test sigmoid with zero"""
        input_value = 0
        expected_output = 1 / (1 + np.exp(-input_value))
        self.assertAlmostEqual(sigmoid(input_value), expected_output, places=6)
    
    def test_sigmoid_scalar(self):
        """Test sigmoid with a scalar"""
        input_value = 0.5
        expected_output = 1 / (1 + np.exp(-input_value))
        self.assertAlmostEqual(sigmoid(input_value), expected_output, places=6)

if __name__ == '__main__':
    unittest.main()
