import unittest
import pandas as pd

import sys
import os

# Adjust the import path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import the detect_outliers_iqr function
from src.preprocessing.validation_functions import detect_outliers_iqr

class TestDetectOutliersIQR(unittest.TestCase):

    def test_detect_outliers_iqr_no_outliers(self):
        """Test detect_outliers_iqr when there are no outliers."""
        data = pd.DataFrame({
            'pm2_5': [10, 15, 20, 25, 30, 35, 40]
        })
        upper_bound = detect_outliers_iqr(data, column='pm2_5')
        Q1 = data['pm2_5'].quantile(0.25)
        Q3 = data['pm2_5'].quantile(0.75)
        IQR = Q3 - Q1
        expected_upper_bound = Q3 + 1.5 * IQR
        self.assertEqual(upper_bound, expected_upper_bound)  # Compare with dynamically calculated value
        outliers = data[data['pm2_5'] > upper_bound]
        self.assertTrue(outliers.empty)  # No outliers expected

    def test_detect_outliers_iqr_with_outliers(self):
        """Test detect_outliers_iqr when there are outliers."""
        data = pd.DataFrame({
            'pm2_5': [10, 15, 20, 25, 30, 35, 200]  # 200 is an outlier
        })
        upper_bound = detect_outliers_iqr(data, column='pm2_5')
        Q1 = data['pm2_5'].quantile(0.25)
        Q3 = data['pm2_5'].quantile(0.75)
        IQR = Q3 - Q1
        expected_upper_bound = Q3 + 1.5 * IQR
        self.assertEqual(upper_bound, expected_upper_bound)  # Compare with dynamically calculated value
        outliers = data[data['pm2_5'] > upper_bound]
        self.assertFalse(outliers.empty)
        self.assertEqual(outliers['pm2_5'].values[0], 200)  # Verify that 200 is detected as outlier

    def test_detect_outliers_iqr_all_outliers(self):
        """Test detect_outliers_iqr when all values are above the upper bound."""
        data = pd.DataFrame({
            'pm2_5': [100, 150, 200, 250, 300]
        })
        upper_bound = detect_outliers_iqr(data, column='pm2_5')
        Q1 = data['pm2_5'].quantile(0.25)
        Q3 = data['pm2_5'].quantile(0.75)
        IQR = Q3 - Q1
        expected_upper_bound = Q3 + 1.5 * IQR
        outliers = data[data['pm2_5'] > upper_bound]
        self.assertEqual(outliers.shape[0], 0)  # There should be no outliers, upper bound is high for this data

    def test_detect_outliers_iqr_with_custom_column(self):
        """Test detect_outliers_iqr with a custom column name."""
        data = pd.DataFrame({
            'custom_pm2_5': [10, 15, 20, 25, 30, 35, 40, 200]  # 200 is an outlier
        })
        upper_bound = detect_outliers_iqr(data, column='custom_pm2_5')
        Q1 = data['custom_pm2_5'].quantile(0.25)
        Q3 = data['custom_pm2_5'].quantile(0.75)
        IQR = Q3 - Q1
        expected_upper_bound = Q3 + 1.5 * IQR
        self.assertEqual(upper_bound, expected_upper_bound)  # Compare with dynamically calculated value
        outliers = data[data['custom_pm2_5'] > upper_bound]
        self.assertFalse(outliers.empty)
        self.assertEqual(outliers['custom_pm2_5'].values[0], 200)  # Verify that 200 is detected as outlier

if __name__ == '__main__':
    unittest.main()