import unittest
import numpy as np
from sklearn.metrics import mean_squared_error

import sys
import os

# Adjust the import path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import the rmse_scorer function
from src.evaluation.evaluation_functions import rmse_scorer

class TestRMSEScorer(unittest.TestCase):
    
    def test_rmse_scorer_perfect_prediction(self):
        """Test RMSE when predictions are perfect."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        expected_rmse = 0.0  # No error, so RMSE should be 0
        actual_rmse = rmse_scorer(y_true, y_pred)
        self.assertAlmostEqual(actual_rmse, expected_rmse, places=6)
    
    def test_rmse_scorer_with_errors(self):
        """Test RMSE when there are prediction errors."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([2, 2, 3, 5, 5])  # Some predictions are off
        expected_rmse = np.sqrt(mean_squared_error(y_true, y_pred))  # Calculate expected RMSE
        actual_rmse = rmse_scorer(y_true, y_pred)
        self.assertAlmostEqual(actual_rmse, expected_rmse, places=6)
    
    def test_rmse_scorer_large_values(self):
        """Test RMSE with large values."""
        y_true = np.array([1000, 2000, 3000])
        y_pred = np.array([1100, 1900, 2900])
        expected_rmse = np.sqrt(mean_squared_error(y_true, y_pred))  # Calculate expected RMSE
        actual_rmse = rmse_scorer(y_true, y_pred)
        self.assertAlmostEqual(actual_rmse, expected_rmse, places=6)

    def test_rmse_scorer_single_value(self):
        """Test RMSE with a single value."""
        y_true = np.array([100])
        y_pred = np.array([105])
        expected_rmse = np.sqrt(mean_squared_error(y_true, y_pred))  # Calculate expected RMSE
        actual_rmse = rmse_scorer(y_true, y_pred)
        self.assertAlmostEqual(actual_rmse, expected_rmse, places=6)

if __name__ == '__main__':
    unittest.main()