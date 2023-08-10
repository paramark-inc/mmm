import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import os
import sys
import unittest

from datetime import date, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "impl", "lightweight_mmm"))

from mmm.constants import constants
from mmm.data import InputData, DataToFit


class ModelTestCase(unittest.TestCase):
    @staticmethod
    def _generate_test_input_data(observations=100):
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
            media_priors=media_costs,
            media_names=["Channel1", "Channel2"],
            extra_features_data=extra_features_data,
            extra_features_names=["Feature1", "Feature2", "Feature3"],
            target_data=target_data,
            target_is_log_scale=False,
            target_name="Target",
        )

    # noinspection PyMethodMayBeStatic
    def test_make_data_to_fit(self):
        input_data = ModelTestCase._generate_test_input_data()
        data_to_fit = DataToFit.from_input_data(input_data)
        assert_array_equal(data_to_fit.date_strs, input_data.date_strs)

        data_to_fit_media_data = np.vstack(
            (data_to_fit.media_data_train_scaled, data_to_fit.media_data_test_scaled)
        )
        data_to_fit_media_data_unscaled = data_to_fit.media_scaler.inverse_transform(
            data_to_fit_media_data
        )
        assert_array_almost_equal(data_to_fit_media_data_unscaled, input_data.media_data, decimal=3)

        data_to_fit_media_costs_unscaled = data_to_fit.media_costs_scaler.inverse_transform(
            data_to_fit.media_costs_scaled
        )
        assert_array_almost_equal(
            data_to_fit_media_costs_unscaled, input_data.media_costs, decimal=3
        )

        data_to_fit_extra_features_data = np.vstack(
            (data_to_fit.extra_features_train_scaled, data_to_fit.extra_features_test_scaled)
        )
        data_to_fit_extra_features_data_unscaled = (
            data_to_fit.extra_features_scaler.inverse_transform(data_to_fit_extra_features_data)
        )
        assert_array_almost_equal(
            data_to_fit_extra_features_data_unscaled, input_data.extra_features_data, decimal=3
        )

        data_to_fit_target_data = np.hstack(
            (data_to_fit.target_train_scaled, data_to_fit.target_test_scaled)
        )
        data_to_fit_target_data_unscaled = data_to_fit.target_scaler.inverse_transform(
            data_to_fit_target_data
        )
        assert_array_almost_equal(
            data_to_fit_target_data_unscaled, input_data.target_data, decimal=3
        )

    def test_to_data_frame(self):
        input_data = ModelTestCase._generate_test_input_data()

        data_to_fit = DataToFit.from_input_data(input_data)
        data_to_fit_media_data = np.vstack(
            (data_to_fit.media_data_train_scaled, data_to_fit.media_data_test_scaled)
        )
        data_to_fit_costs_by_row_data = np.vstack(
            (
                data_to_fit.media_costs_by_row_train_scaled,
                data_to_fit.media_costs_by_row_test_scaled,
            )
        )

        data_to_fit_extra_features_data = np.vstack(
            (data_to_fit.extra_features_train_scaled, data_to_fit.extra_features_test_scaled)
        )
        data_to_fit_target_data = np.hstack(
            (data_to_fit.target_train_scaled, data_to_fit.target_test_scaled)
        )

        # unscaled = False
        per_observation_df, per_channel_df = data_to_fit.to_data_frame()

        self.assertEqual(100, per_observation_df.shape[0])
        self.assertEqual(2 + 2 + 3 + 1, per_observation_df.shape[1])

        assert_array_almost_equal(
            per_observation_df["Channel1 volume"], data_to_fit_media_data[:, 0], decimal=3
        )
        assert_array_almost_equal(
            per_observation_df["Channel2 volume"], data_to_fit_media_data[:, 1], decimal=3
        )
        assert_array_almost_equal(
            per_observation_df["Channel1 cost"], data_to_fit_costs_by_row_data[:, 0], decimal=3
        )
        assert_array_almost_equal(
            per_observation_df["Channel2 cost"], data_to_fit_costs_by_row_data[:, 1], decimal=3
        )

        self.assertAlmostEqual(
            data_to_fit_media_data[1, 1], per_observation_df.loc["2022-01-02", "Channel2 volume"]
        )

        assert_array_almost_equal(
            per_observation_df["Feature1"], data_to_fit_extra_features_data[:, 0], decimal=3
        )
        assert_array_almost_equal(
            per_observation_df["Feature2"], data_to_fit_extra_features_data[:, 1], decimal=3
        )
        assert_array_almost_equal(
            per_observation_df["Feature3"], data_to_fit_extra_features_data[:, 2], decimal=3
        )
        assert_array_almost_equal(per_observation_df["Target"], data_to_fit_target_data, decimal=3)

        self.assertEqual(2, per_channel_df.shape[0])
        assert_array_almost_equal(per_channel_df["Cost"], data_to_fit.media_costs_scaled, decimal=3)

        # unscaled = True
        per_observation_df, per_channel_df = data_to_fit.to_data_frame(unscaled=True)

        self.assertEqual(100, per_observation_df.shape[0])
        self.assertEqual(2 + 2 + 3 + 1, per_observation_df.shape[1])

        assert_array_almost_equal(
            per_observation_df["Channel1 volume"], input_data.media_data[:, 0], decimal=3
        )
        assert_array_almost_equal(
            per_observation_df["Channel2 volume"], input_data.media_data[:, 1], decimal=3
        )
        assert_array_almost_equal(
            per_observation_df["Channel1 cost"], input_data.media_costs_by_row[:, 0], decimal=2
        )
        assert_array_almost_equal(
            per_observation_df["Channel2 cost"], input_data.media_costs_by_row[:, 1], decimal=2
        )

        self.assertAlmostEqual(
            input_data.media_data[1, 1],
            per_observation_df.loc["2022-01-02", "Channel2 volume"],
            places=3,
        )

        assert_array_almost_equal(
            per_observation_df["Feature1"], input_data.extra_features_data[:, 0], decimal=3
        )
        assert_array_almost_equal(
            per_observation_df["Feature2"], input_data.extra_features_data[:, 1], decimal=3
        )
        assert_array_almost_equal(
            per_observation_df["Feature3"], input_data.extra_features_data[:, 2], decimal=3
        )
        assert_array_almost_equal(per_observation_df["Target"], input_data.target_data, decimal=3)

        self.assertEqual(2, per_channel_df.shape[0])
        assert_array_almost_equal(per_channel_df["Cost"], input_data.media_costs, decimal=3)

    def test_clone_as_weekly(self):
        # un-even number of observations
        input_data = ModelTestCase._generate_test_input_data(observations=100)
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
        input_data = ModelTestCase._generate_test_input_data(observations=98)
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
        input_data = ModelTestCase._generate_test_input_data(observations=100)

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
        input_data = ModelTestCase._generate_test_input_data(observations=100)

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
