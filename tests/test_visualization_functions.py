import unittest
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import sys
import os

# Adjust the import path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import the visualization function
from src.visualization.visualization_functions import (
    plot_box_plot, plot_violin_plot, plot_histogram, 
    plot_box_plots, plot_violin_plots, plot_histograms, 
    plot_correlation_matrix_heatmap, plot_scatter_plots, error_analysis_plot
)

class TestVisualizationFunctions(unittest.TestCase):

    def setUp(self):
        """Set up test data for use in all test cases."""
        # Generate sample DataFrame for testing
        self.data = pd.DataFrame({
            'value': [10, 20, 30, 40, 50],
            'category': ['A', 'B', 'A', 'B', 'A'],
            'other_value': [5, 15, 25, 35, 45],
            'date': pd.date_range('20230101', periods=5)
        })
        
        # Generate correlation matrix using only numeric columns
        self.corr_matrix = self.data[['value', 'other_value']].corr()

    def tearDown(self):
        """Tear down test plots to avoid overlapping tests."""
        plt.close('all')

    def test_plot_box_plot(self):
        """Test that the plot_box_plot function runs without errors."""
        try:
            plot_box_plot(self.data, value_col='value', category_col='category')
        except Exception as e:
            self.fail(f"plot_box_plot raised an exception: {e}")

    def test_plot_violin_plot(self):
        """Test that the plot_violin_plot function runs without errors."""
        try:
            plot_violin_plot(self.data, value_col='value', category_col='category')
        except Exception as e:
            self.fail(f"plot_violin_plot raised an exception: {e}")

    def test_plot_histogram(self):
        """Test that the plot_histogram function runs without errors."""
        try:
            plot_histogram(self.data, column='value')
        except Exception as e:
            self.fail(f"plot_histogram raised an exception: {e}")

    def test_plot_box_plots(self):
        """Test that the plot_box_plots function runs without errors."""
        try:
            plot_box_plots(self.data, columns_to_plot=['value', 'other_value'])
        except Exception as e:
            self.fail(f"plot_box_plots raised an exception: {e}")

    def test_plot_violin_plots(self):
        """Test that the plot_violin_plots function runs without errors."""
        try:
            plot_violin_plots(self.data, columns_to_plot=['value', 'other_value'])
        except Exception as e:
            self.fail(f"plot_violin_plots raised an exception: {e}")

    def test_plot_histograms(self):
        """Test that the plot_histograms function runs without errors."""
        try:
            plot_histograms(self.data, columns_to_plot=['value', 'other_value'])
        except Exception as e:
            self.fail(f"plot_histograms raised an exception: {e}")

    def test_plot_correlation_matrix_heatmap(self):
        """Test that the plot_correlation_matrix_heatmap function runs without errors."""
        try:
            plot_correlation_matrix_heatmap(self.corr_matrix)
        except Exception as e:
            self.fail(f"plot_correlation_matrix_heatmap raised an exception: {e}")

    def test_plot_scatter_plots(self):
        """Test that the plot_scatter_plots function runs without errors."""
        try:
            plot_scatter_plots(self.data, columns_to_plot=['value', 'other_value'], target_variable='value')
        except Exception as e:
            self.fail(f"plot_scatter_plots raised an exception: {e}")

    def test_error_analysis_plot(self):
        """Test that the error_analysis_plot function runs without errors."""
        y_test = np.array([10, 20, 30, 40, 50])
        y_pred_test = np.array([12, 18, 29, 39, 52])
        try:
            error_analysis_plot(y_test, y_pred_test)
        except Exception as e:
            self.fail(f"error_analysis_plot raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()