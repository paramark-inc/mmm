import numpy as np
import pandas as pd
import unittest
from numpy.testing import assert_array_almost_equal

from mmm.data import DataToFit, InputData
from mmm.describe.describe import _get_summary_df
from .test_input_data import InputDataTest


class DescribeTest(unittest.TestCase):
    def setUp(self):
        """Set up test data for each test method."""
        self.input_data = InputDataTest.generate_test_input_data(observations=10)
        self.data_to_fit = DataToFit.from_input_data(input_data=self.input_data, config={})
        
        # Create geo input data with 3D structure: [time, channel, geo]
        self.geo_input_data = InputData(
            date_strs=np.array(["2022-01-01", "2022-01-02"]),
            time_granularity="daily",
            media_data=np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),  # [time, channel, geo]
            media_costs=np.array([10.0, 20.0]),  # [channel]
            media_costs_by_row=np.array([[[5.0, 10.0], [5.0, 10.0]], [[5.0, 10.0], [5.0, 10.0]]]),  # [time, channel, geo]
            media_cost_priors=np.array([10.0, 20.0]),  # [channel]
            learned_media_priors=np.zeros(shape=(2,)),
            media_names=["Channel1", "Channel2"],
            extra_features_data=np.array([[[1.0, 2.0]], [[3.0, 4.0]]]),  # [time, feature, geo]
            extra_features_names=["Feature1"],
            target_data=np.array([[100.0, 200.0], [300.0, 400.0]]),  # [time, geo]
            target_is_log_scale=False,
            target_name="Target",
            geo_names=["Geo1", "Geo2"],
        )
        
        self.geo_data_to_fit = DataToFit.from_input_data(input_data=self.geo_input_data, config={})

    def test_get_summary_df_global_model(self):
        """Test _get_summary_df with a global model (no geo data)."""
        # Create test data: 100 samples, 2 channels
        media_distributions = np.random.normal(0, 1, (100, 2))
        blended_distribution = np.random.normal(0, 1, (100,))
        
        result = _get_summary_df(
            data_to_fit=self.data_to_fit,
            media_distributions=media_distributions,
            blended_distribution=blended_distribution,
            geo_name=None
        )
        
        # Check that result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)
        
        # Check expected shape: 3 rows (blended + 2 channels), columns for quantiles + median + mean
        expected_columns = ["0.05", "0.95", "median", "mean"]  # Default CI quantiles for 0.9 credibility interval
        self.assertEqual(result.shape, (3, 4))
        self.assertEqual(list(result.columns), expected_columns)
        self.assertEqual(list(result.index), ["blended", "Channel1", "Channel2"])
        
        # Check that blended values match the blended_distribution
        expected_blended_quantiles = np.quantile(blended_distribution, [0.05, 0.95])
        assert_array_almost_equal(result.loc["blended", ["0.05", "0.95"]], expected_blended_quantiles)
        self.assertAlmostEqual(result.loc["blended", "median"], np.median(blended_distribution))
        self.assertAlmostEqual(result.loc["blended", "mean"], np.mean(blended_distribution))
        
        # Check that channel values match the media_distributions
        for i, channel_name in enumerate(["Channel1", "Channel2"]):
            expected_channel_quantiles = np.quantile(media_distributions[:, i], [0.05, 0.95])
            assert_array_almost_equal(result.loc[channel_name, ["0.05", "0.95"]], expected_channel_quantiles)
            self.assertAlmostEqual(result.loc[channel_name, "median"], np.median(media_distributions[:, i]))
            self.assertAlmostEqual(result.loc[channel_name, "mean"], np.mean(media_distributions[:, i]))

    def test_get_summary_df_geo_model(self):
        """Test _get_summary_df with a geo model."""
        # Create test data: 100 samples, 2 channels, 2 geos
        media_distributions = np.random.normal(0, 1, (100, 2, 2))
        blended_distribution = np.random.normal(0, 1, (100, 2))
        
        # Test with specific geo
        result = _get_summary_df(
            data_to_fit=self.geo_data_to_fit,
            media_distributions=media_distributions,
            blended_distribution=blended_distribution,
            geo_name="Geo1"
        )
        
        # Check that result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)
        
        # Check expected shape: 3 rows (blended + 2 channels), columns for quantiles + median + mean
        expected_columns = ["0.05", "0.95", "median", "mean"]
        self.assertEqual(result.shape, (3, 4))
        self.assertEqual(list(result.columns), expected_columns)
        self.assertEqual(list(result.index), ["blended", "Channel1", "Channel2"])
        
        # Check that values match the data for Geo1 (index 0)
        geo_idx = 0
        expected_blended_quantiles = np.quantile(blended_distribution[:, geo_idx], [0.05, 0.95])
        assert_array_almost_equal(result.loc["blended", ["0.05", "0.95"]], expected_blended_quantiles)
        self.assertAlmostEqual(result.loc["blended", "median"], np.median(blended_distribution[:, geo_idx]))
        self.assertAlmostEqual(result.loc["blended", "mean"], np.mean(blended_distribution[:, geo_idx]))
        
        # Check that channel values match the media_distributions for Geo1
        for i, channel_name in enumerate(["Channel1", "Channel2"]):
            expected_channel_quantiles = np.quantile(media_distributions[:, i, geo_idx], [0.05, 0.95])
            assert_array_almost_equal(result.loc[channel_name, ["0.05", "0.95"]], expected_channel_quantiles)
            self.assertAlmostEqual(result.loc[channel_name, "median"], np.median(media_distributions[:, i, geo_idx]))
            self.assertAlmostEqual(result.loc[channel_name, "mean"], np.mean(media_distributions[:, i, geo_idx]))

    def test_get_summary_df_invalid_geo_name(self):
        """Test _get_summary_df with an invalid geo name."""
        # Create test data
        media_distributions = np.random.normal(0, 1, (100, 2, 2))
        blended_distribution = np.random.normal(0, 1, (100, 2))
        
        # Test with invalid geo name
        with self.assertRaises(ValueError) as context:
            _get_summary_df(
                data_to_fit=self.geo_data_to_fit,
                media_distributions=media_distributions,
                blended_distribution=blended_distribution,
                geo_name="InvalidGeo"
            )
        
        self.assertIn("is not in list", str(context.exception))

    def test_get_summary_df_geo_name_for_global_model(self):
        """Test _get_summary_df with geo_name passed for a global model."""
        media_distributions = np.random.normal(0, 1, (100, 2))
        blended_distribution = np.random.normal(0, 1, (100,))
        
        with self.assertRaises(ValueError) as context:
            _get_summary_df(
                data_to_fit=self.data_to_fit,  # Global model (no geo_names)
                media_distributions=media_distributions,
                blended_distribution=blended_distribution,
                geo_name="SomeGeo"  # This should cause an error
            )
        
        self.assertIn("geo_name passed for a global model", str(context.exception))

    def test_get_summary_df_no_geo_name_for_geo_model(self):
        """Test _get_summary_df with no geo_name passed for a geo model."""
        # Create test data
        media_distributions = np.random.normal(0, 1, (100, 2, 2))
        blended_distribution = np.random.normal(0, 1, (100, 2))
        
        with self.assertRaises(ValueError) as context:
            _get_summary_df(
                data_to_fit=self.geo_data_to_fit,  # Geo model (has geo_names)
                media_distributions=media_distributions,
                blended_distribution=blended_distribution,
                geo_name=None  # This should cause an error
            )
        
        self.assertIn("summarizing across geos is not supported", str(context.exception))

    def test_get_summary_df_wrong_dimensions(self):
        """Test _get_summary_df with wrong array dimensions."""
        # Test with 3D media_distributions for global model (should be 2D)
        media_distributions = np.random.normal(0, 1, (100, 2, 1))  # Wrong: 3D for global model
        blended_distribution = np.random.normal(0, 1, (100,))
        
        with self.assertRaises(AssertionError):
            _get_summary_df(
                data_to_fit=self.data_to_fit,  # Global model
                media_distributions=media_distributions,
                blended_distribution=blended_distribution,
                geo_name=None
            )

    def test_get_summary_df_custom_ci_quantiles(self):
        """Test _get_summary_df with custom CI quantiles."""
        # Mock the get_ci_quantiles method to return custom quantiles
        original_method = self.data_to_fit.get_ci_quantiles
        self.data_to_fit.get_ci_quantiles = lambda: [0.1, 0.9]
        
        try:
            media_distributions = np.random.normal(0, 1, (100, 2))
            blended_distribution = np.random.normal(0, 1, (100,))
            
            result = _get_summary_df(
                data_to_fit=self.data_to_fit,
                media_distributions=media_distributions,
                blended_distribution=blended_distribution,
                geo_name=None
            )
            
            # Check that custom quantiles are used
            expected_columns = ["0.1", "0.9", "median", "mean"]
            self.assertEqual(list(result.columns), expected_columns)
            
            # Check that values match the custom quantiles
            expected_blended_quantiles = np.quantile(blended_distribution, [0.1, 0.9])
            assert_array_almost_equal(result.loc["blended", ["0.1", "0.9"]], expected_blended_quantiles)
            
        finally:
            # Restore original method
            self.data_to_fit.get_ci_quantiles = original_method

    def test_get_summary_df_edge_cases(self):
        """Test _get_summary_df with edge cases like single sample."""
        # Test with single sample
        media_distributions = np.array([[1.0, 2.0]])  # 1 sample, 2 channels
        blended_distribution = np.array([3.0])  # 1 sample
        
        result = _get_summary_df(
            data_to_fit=self.data_to_fit,
            media_distributions=media_distributions,
            blended_distribution=blended_distribution,
            geo_name=None
        )
        
        # With single sample, quantiles should equal the single value
        self.assertAlmostEqual(result.loc["blended", "0.05"], 3.0)
        self.assertAlmostEqual(result.loc["blended", "0.95"], 3.0)
        self.assertAlmostEqual(result.loc["blended", "median"], 3.0)
        self.assertAlmostEqual(result.loc["blended", "mean"], 3.0)
        
        self.assertAlmostEqual(result.loc["Channel1", "0.05"], 1.0)
        self.assertAlmostEqual(result.loc["Channel1", "0.95"], 1.0)
        self.assertAlmostEqual(result.loc["Channel1", "median"], 1.0)
        self.assertAlmostEqual(result.loc["Channel1", "mean"], 1.0)
        
        self.assertAlmostEqual(result.loc["Channel2", "0.05"], 2.0)
        self.assertAlmostEqual(result.loc["Channel2", "0.95"], 2.0)
        self.assertAlmostEqual(result.loc["Channel2", "median"], 2.0)
        self.assertAlmostEqual(result.loc["Channel2", "mean"], 2.0)
