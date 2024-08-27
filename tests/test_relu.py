import unittest
import numpy as np
from faulkner.activation import reLU

class TestReLU(unittest.TestCase):
    
    def test_relu_positive_values(self):
        """Test ReLU with positive values"""
        input_array = np.array([1, 2, 3])
        expected_output = np.array([1, 2, 3])
        np.testing.assert_array_equal(reLU(input_array), expected_output)
    
    def test_relu_negative_values(self):
        """Test ReLU with negative values"""
        input_array = np.array([-1, -2, -3])
        expected_output = np.array([0, 0, 0])
        np.testing.assert_array_equal(reLU(input_array), expected_output)
    
    def test_relu_mixed_values(self):
        """Test ReLU with mixed positive and negative values"""
        input_array = np.array([-1, 0, 1, 2])
        expected_output = np.array([0, 0, 1, 2])
        np.testing.assert_array_equal(reLU(input_array), expected_output)
    
    def test_relu_zero(self):
        """Test ReLU with zero"""
        input_array = np.array([0])
        expected_output = np.array([0])
        np.testing.assert_array_equal(reLU(input_array), expected_output)

if __name__ == '__main__':
    unittest.main()
