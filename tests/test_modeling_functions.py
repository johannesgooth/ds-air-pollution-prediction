import unittest
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
import mlflow

import sys
import os

# Adjust the import path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))  # Go up two levels to the project root
sys.path.insert(0, project_root)  # Add the project root to sys.path

# Corrected import path
from src.modeling.modeling_functions import train_and_evaluate_model

class TestModelingFunctions(unittest.TestCase):

    def setUp(self):
        """Set up test data for use in all test cases."""
        # Sample training data
        self.X_train = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10]
        })
        self.y_train = pd.Series([1, 2, 3, 4, 5])
        
        # Sample testing data
        self.X_test = pd.DataFrame({
            'feature1': [6, 7],
            'feature2': [12, 14]
        })
        self.y_test = pd.Series([6, 7])
        
        # Sample preprocessing pipeline
        self.preprocessing_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=1))
        ])
        
        # Parameter grid for tuning
        self.param_grid = {
            'model__fit_intercept': [True, False]
        }

    def test_train_and_evaluate_model(self):
        """Test that train_and_evaluate_model runs without errors and returns expected outputs."""
        
        # Run the function and unpack three returned values
        best_model, test_results, y_test_pred = train_and_evaluate_model(
            model=LinearRegression(),
            param_grid=self.param_grid,
            preprocessing_pipeline=self.preprocessing_pipeline,
            search_type='grid',
            X_train=self.X_train,
            X_test=self.X_test,
            y_train=self.y_train,
            y_test=self.y_test
        )
        
        # Check that the best_model is an instance of Pipeline
        self.assertIsInstance(best_model, Pipeline, "best_model should be a Pipeline instance")
        
        # Check that test_results is a dictionary containing expected keys
        self.assertIsInstance(test_results, dict, "test_results should be a dictionary")
        self.assertIn('rmse_test', test_results, "test_results should contain 'rmse_test'")
        self.assertIn('r2_test', test_results, "test_results should contain 'r2_test'")
        self.assertIn('adj_r2_test', test_results, "test_results should contain 'adj_r2_test'")
        
        # Check that y_test_pred is a NumPy array or pandas Series with correct length
        self.assertTrue(
            isinstance(y_test_pred, (pd.Series, pd.DataFrame, pd.Index)) or 
            hasattr(y_test_pred, '__len__'),
            "y_test_pred should be a pandas Series, DataFrame, Index, or have a length"
        )
        self.assertEqual(
            len(y_test_pred), 
            len(self.y_test), 
            "y_test_pred should have the same length as y_test"
        )
        
        # Optional: Verify that predictions are reasonable (e.g., within expected range)
        # This can be adjusted based on domain knowledge
        self.assertTrue(
            all(self.y_test.min() - 10 <= y_test_pred) and all(y_test_pred <= self.y_test.max() + 10),
            "Predicted values should be within a reasonable range of actual values"
        )

if __name__ == '__main__':
    unittest.main()