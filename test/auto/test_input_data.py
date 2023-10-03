import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import unittest

from datetime import date, timedelta

from mmm.constants import constants
from mmm.data import InputData, DataToFit


class InputDataTest(unittest.TestCase):
    @staticmethod
    def generate_test_input_data(observations=100):
        date_strs = [
            (date.fromisoformat("2022-01-01") + timedelta(days=idx)).strftime("%-m/%-d/%Y")
            for idx in range(observations)
        ]

        # n observations * 2 channels
        media_data = np.arange(observations * 2).astype(np.float64).reshape((observations, 2))
        media_costs_by_row = media_data.copy()
        media_costs_by_row[:, 0] *= 100.0
        media_costs_by_row[:, 1] *= 200.0
        media_costs = media_costs_by_row.sum(axis=0)

        # n observations * 3 features
        extra_features_data = (np.arange(observations * 3).astype(np.float64) * 2.0).reshape(
            (observations, 3)
        )

        target_data = np.arange(observations).astype(np.float64) * 100.0

        return InputData(
            date_strs=np.array(date_strs),
            time_granularity=constants.GRANULARITY_DAILY,
            media_data=media_data,
            media_costs=media_costs,
            media_costs_by_row=media_costs_by_row,
            media_cost_priors=media_costs,
            learned_media_priors=np.zeros(shape=media_costs.shape),
            media_names=["Channel1", "Channel2"],
            extra_features_data=extra_features_data,
            extra_features_names=["Feature1", "Feature2", "Feature3"],
            target_data=target_data,
            target_is_log_scale=False,
            target_name="Target",
        )

    def test_input_data_trivial_create(self):
        input_data = InputData(
            date_strs=np.array(["1999-12-31"]),
            time_granularity=constants.GRANULARITY_DAILY,
            media_data=np.array([[1.0]]),
            media_costs=np.array([3.0]),
            media_costs_by_row=np.array([[3.0]]),
            media_cost_priors=np.array([3.0]),
            learned_media_priors=np.zeros(shape=(1,)),
            media_names=["foo"],
            extra_features_data=np.array([[4.0]]),
            extra_features_names=["bar"],
            target_data=np.array([9.0]),
            target_is_log_scale=False,
            target_name="baz",
        )

        self.assertIsInstance(input_data, InputData)

    def test_clone_as_weekly(self):
        # un-even number of observations
        input_data = InputDataTest.generate_test_input_data(observations=100)
        input_data_weekly = input_data.clone_as_weekly()

        self.assertEqual(14, input_data_weekly.date_strs.shape[0])
        self.assertEqual(constants.GRANULARITY_WEEKLY, input_data_weekly.time_granularity)
        self.assertEqual(14, input_data_weekly.media_data.shape[0])
        self.assertEqual(14, input_data_weekly.media_costs_by_row.shape[0])
        assert_array_almost_equal(
            input_data.media_data[0:98, :].sum(axis=0), input_data_weekly.media_data.sum(axis=0)
        )
        assert_array_almost_equal(
            input_data.media_costs_by_row[0:98, :].sum(axis=0),
            input_data_weekly.media_costs_by_row.sum(axis=0),
        )
        assert_array_equal(input_data.media_costs, input_data_weekly.media_costs)
        assert_array_equal(input_data.media_names, input_data_weekly.media_names)
        self.assertEqual(14, input_data_weekly.extra_features_data.shape[0])
        assert_array_almost_equal(
            input_data.extra_features_data[0:98, :].sum(axis=0),
            input_data_weekly.extra_features_data.sum(axis=0),
        )
        assert_array_equal(input_data.extra_features_names, input_data_weekly.extra_features_names)
        self.assertEqual(14, input_data_weekly.target_data.shape[0])
        self.assertAlmostEqual(
            input_data.target_data[0:98].sum(), input_data_weekly.target_data.sum()
        )

        # even number of observations
        input_data = InputDataTest.generate_test_input_data(observations=98)
        input_data_weekly = input_data.clone_as_weekly()

        self.assertEqual(14, input_data_weekly.date_strs.shape[0])
        self.assertEqual(constants.GRANULARITY_WEEKLY, input_data_weekly.time_granularity)
        self.assertEqual(14, input_data_weekly.media_data.shape[0])
        self.assertEqual(14, input_data_weekly.media_costs_by_row.shape[0])
        assert_array_almost_equal(
            input_data.media_data.sum(axis=0), input_data_weekly.media_data.sum(axis=0)
        )
        assert_array_almost_equal(
            input_data.media_costs_by_row.sum(axis=0),
            input_data_weekly.media_costs_by_row.sum(axis=0),
        )
        assert_array_equal(input_data.media_costs, input_data_weekly.media_costs)
        assert_array_equal(input_data.media_names, input_data_weekly.media_names)
        self.assertEqual(14, input_data_weekly.extra_features_data.shape[0])
        assert_array_almost_equal(
            input_data.extra_features_data.sum(axis=0),
            input_data_weekly.extra_features_data.sum(axis=0),
        )
        assert_array_equal(input_data.extra_features_names, input_data_weekly.extra_features_names)
        self.assertEqual(14, input_data_weekly.target_data.shape[0])
        self.assertAlmostEqual(input_data.target_data.sum(), input_data_weekly.target_data.sum())

    def test_clone_and_add_extra_features(self):
        input_data = InputDataTest.generate_test_input_data(observations=100)

        # 100 observations * 2 features
        new_feature_data = (np.arange(100 * 2).astype(np.float64) * 30.0).reshape(100, 2)
        new_input_data = input_data.clone_and_add_extra_features(
            ["NewFeature1", "NewFeature2"], new_feature_data
        )

        self.assertEqual(
            ["Feature1", "Feature2", "Feature3", "NewFeature1", "NewFeature2"],
            new_input_data.extra_features_names,
        )
        assert_array_almost_equal(new_feature_data, new_input_data.extra_features_data[:, 3:5])

    def test_clone_and_split_media_data(self):
        input_data = InputDataTest.generate_test_input_data(observations=100)

        input_data_split_channel_0 = input_data.clone_and_split_media_data(
            channel_idx=0,
            split_obs_idx=10,
            media_before_name="Channel1 (before)",
            media_after_name="Channel1 (after)",
        )

        self.assertEqual(
            input_data.media_data.shape[0], input_data_split_channel_0.media_data.shape[0]
        )
        self.assertEqual(3, input_data_split_channel_0.media_data.shape[1])
        self.assertAlmostEqual(0.0, input_data_split_channel_0.media_data[10:, 0].sum())
        self.assertAlmostEqual(
            input_data.media_data[:10, 0].sum(), input_data_split_channel_0.media_data[:10, 0].sum()
        )
        self.assertAlmostEqual(0.0, input_data_split_channel_0.media_data[:10, 1].sum())
        self.assertAlmostEqual(
            input_data.media_data[10:, 0].sum(), input_data_split_channel_0.media_data[10:, 1].sum()
        )
        self.assertAlmostEqual(
            input_data.media_data[:, 1].sum(), input_data_split_channel_0.media_data[:, 2].sum()
        )

        input_data_split_channel_1 = input_data.clone_and_split_media_data(
            channel_idx=1,
            split_obs_idx=input_data.media_data.shape[0] - 1,
            media_before_name="Channel2 (before)",
            media_after_name="Channel2 (after)",
        )

        self.assertEqual(
            input_data.media_data.shape[0], input_data_split_channel_1.media_data.shape[0]
        )
        self.assertEqual(3, input_data_split_channel_1.media_data.shape[1])
        self.assertAlmostEqual(0.0, input_data_split_channel_1.media_data[-1, 1])
        self.assertAlmostEqual(
            input_data.media_data[:-1, 1].sum(), input_data_split_channel_1.media_data[:, 1].sum()
        )
        self.assertAlmostEqual(0.0, input_data_split_channel_1.media_data[:-1, 2].sum())
        self.assertAlmostEqual(
            input_data.media_data[:, 0].sum(), input_data_split_channel_1.media_data[:, 0].sum()
        )


if __name__ == "__main__":
    unittest.main()
