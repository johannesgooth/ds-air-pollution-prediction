import unittest
import pandas as pd
import numpy as np
from datetime import datetime

import sys
import os

# Adjust the import path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import the transformer classes
from src.preprocessing.transformer_classes import (
    DropUvaerosolLayerHeightTransformer,
    ConvertDateTransformer, SortByDateTransformer,
    ExtractDateComponentsTransformer, DropDateTransformer
)

class TestTransformers(unittest.TestCase):
    
    def test_drop_uvaerosol_layer_height_transformer(self):
        data = pd.DataFrame({
            'uvaerosollayerheight': [100, 200, 300],
            'other_col': [10, 20, 30]
        })
        transformer = DropUvaerosolLayerHeightTransformer()
        transformed = transformer.transform(data)
        self.assertNotIn('uvaerosollayerheight', transformed.columns)
        self.assertIn('other_col', transformed.columns)
    
    def test_convert_date_transformer(self):
        data = pd.DataFrame({
            'date': ['2023-01-01', '2023-02-01', 'invalid-date'],
            'value': [10, 20, 30]
        })
        transformer = ConvertDateTransformer(date_column='date', date_format='%Y-%m-%d')
        transformed = transformer.transform(data)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(transformed['date']))
        self.assertEqual(transformed['date'].isnull().sum(), 1)  # 1 invalid date

    def test_sort_by_date_transformer(self):
        data = pd.DataFrame({
            'date': pd.to_datetime(['2023-03-01', '2023-01-01', '2023-02-01']),
            'value': [30, 10, 20]
        })
        transformer = SortByDateTransformer(date_column='date')
        transformed = transformer.transform(data)
        self.assertTrue(transformed['date'].is_monotonic_increasing)

    def test_extract_date_components_transformer(self):
        data = pd.DataFrame({
            'date': pd.to_datetime(['2023-01-01', '2023-02-01', '2023-03-01']),
            'value': [10, 20, 30]
        })
        transformer = ExtractDateComponentsTransformer(date_column='date')
        transformed = transformer.transform(data)
        self.assertIn('day', transformed.columns)
        self.assertIn('day_of_week', transformed.columns)
        self.assertEqual(transformed.loc[0, 'day'], 1)
        self.assertEqual(transformed.loc[0, 'day_of_week'], 6)  # Sunday (0=Monday, 6=Sunday)

    def test_drop_date_transformer(self):
        data = pd.DataFrame({
            'date': pd.to_datetime(['2023-01-01', '2023-02-01', '2023-03-01']),
            'value': [10, 20, 30]
        })
        transformer = DropDateTransformer(pattern='date')
        transformed = transformer.transform(data)
        self.assertNotIn('date', transformed.columns)
        self.assertIn('value', transformed.columns)


if __name__ == '__main__':
    unittest.main()