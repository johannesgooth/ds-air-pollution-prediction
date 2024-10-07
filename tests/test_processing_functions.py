import unittest
import pandas as pd

import sys
import os

# Adjust the import path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import the processing functions
from src.preprocessing.processing_functions import (
    drop_uvaerosollayerheight_columns, convert_date,
    ordinal_encoding_dates, impute_missing_values, extract_date_components,
    scale_numerical_features
)

class TestProcessingFunctions(unittest.TestCase):

    def test_drop_uvaerosollayerheight_columns(self):
        """Test drop_uvaerosollayerheight_columns function removes correct columns."""
        data = pd.DataFrame({
            'uvaerosollayerheight': [100, 200, 300],
            'other_col': [10, 20, 30]
        })
        result = drop_uvaerosollayerheight_columns(data)
        self.assertNotIn('uvaerosollayerheight', result.columns)
        self.assertIn('other_col', result.columns)

    def test_convert_date(self):
        """Test convert_date function converts dates and handles invalid dates."""
        data = pd.DataFrame({
            'date': ['2023-01-01', 'invalid-date', '2023-03-01'],
            'value': [10, 20, 30]
        })
        result = convert_date(data, date_column='date', date_format='%Y-%m-%d')
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(result['date']))
        self.assertEqual(result['date'].isnull().sum(), 1)  # 1 invalid date

    def test_ordinal_encoding_dates(self):
        """Test ordinal_encoding_dates function sorts by date."""
        data = pd.DataFrame({
            'date': pd.to_datetime(['2023-03-01', '2023-01-01', '2023-02-01']),
            'value': [30, 10, 20]
        })
        result = ordinal_encoding_dates(data)
        self.assertTrue(result['date'].is_monotonic_increasing)

    def test_impute_missing_values(self):
        """Test impute_missing_values function imputes missing values correctly."""
        data = pd.DataFrame({
            'col1': [1.0, 2.0, None],
            'col2': [None, 3.0, 4.0],
            'col3': ['A', 'B', 'C']  # Non-numerical column should be ignored
        })
        result = impute_missing_values(data)
        self.assertFalse(result[['col1', 'col2']].isnull().any().any())  # No missing values in numerical columns

    def test_extract_date_components(self):
        """Test extract_date_components function extracts day and day_of_week."""
        data = pd.DataFrame({
            'date': pd.to_datetime(['2023-01-01', '2023-02-01', '2023-03-01'])
        })
        result = extract_date_components(data)
        self.assertIn('day', result.columns)
        self.assertIn('day_of_week', result.columns)
        self.assertEqual(result.loc[0, 'day'], 1)  # First day of the month
        self.assertEqual(result.loc[0, 'day_of_week'], 6)  # Sunday (0=Monday, 6=Sunday)

    def test_scale_numerical_features(self):
        """Test scale_numerical_features function scales numerical columns correctly."""
        data = pd.DataFrame({
            'col1': [1.0, 2.0, 3.0],
            'col2': [4.0, 5.0, 6.0],
            'date': pd.to_datetime(['2023-01-01', '2023-02-01', '2023-03-01']),
            'target': [10, 20, 30]
        })
        result, numerical_features, scaler = scale_numerical_features(data, exclude_columns=['date', 'target'])
        self.assertIn('col1', numerical_features)
        self.assertIn('col2', numerical_features)
        self.assertNotIn('date', numerical_features)
        self.assertNotIn('target', numerical_features)
        self.assertAlmostEqual(result['col1'].mean(), 0, places=5)  # Check if col1 was scaled (mean should be near 0)

if __name__ == '__main__':
    unittest.main()