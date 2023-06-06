import numpy as np
import unittest

from ...constants import constants

from ...fit.fit import make_data_to_fit
from ...model.model import InputData


class ModelTestCase(unittest.TestCase):

    def test_to_data_frame(self):
        date_strs = np.array(["1/1/2022", "1/2/2022", "1/3/2022"])

        media_data = np.array(
            [[10., 100.],
             [20., 200.],
             [30., 300.]],
            dtype=np.float64
        )

        media_costs = np.array([1000., 5000.], dtype=np.float64)

        extra_features_data = np.array(
            [[40., 400., 4000.],
             [50., 500., 5000.],
             [60., 600., 6000.]],
            dtype=np.float64
        )

        target_data = np.array([900., 1000., 1100.], dtype=np.float64)

        input_data = InputData(
            date_strs=date_strs,
            time_granularity=constants.GRANULARITY_DAILY,
            media_data=media_data,
            media_costs=media_costs,
            media_names=["Channel1", "Channel2"],
            extra_features_data=extra_features_data,
            extra_features_names=["Feature1", "Feature2", "Feature3"],
            target_data=target_data,
            target_name="Target"
        )

        data_to_fit = make_data_to_fit(input_data)
        data_to_fit_media_data = np.vstack(
            (data_to_fit.media_data_train_scaled, data_to_fit.media_data_test_scaled)
        )
        data_to_fit_extra_features_data = np.vstack(
            (data_to_fit.extra_features_train_scaled, data_to_fit.extra_features_test_scaled)
        )
        data_to_fit_target_data = np.hstack((data_to_fit.target_train_scaled, data_to_fit.target_test_scaled))

        per_observation_df, per_channel_df = data_to_fit.to_data_frame()

        self.assertEqual(3, per_observation_df.shape[0])
        self.assertEqual(2 + 3 + 1, per_observation_df.shape[1])
        self.assertAlmostEqual(data_to_fit_media_data[1, 1], per_observation_df.loc["2022-01-02", "Channel2 volume"])
        self.assertAlmostEqual(
            data_to_fit_media_data[:, 0].sum(),
            per_observation_df["Channel1 volume"].sum(),
            places=5
        )
        self.assertAlmostEqual(
            data_to_fit_media_data[:, 1].sum(),
            per_observation_df["Channel2 volume"].sum(),
            places=5
        )
        self.assertAlmostEqual(
            data_to_fit_extra_features_data[:, 0].sum(),
            per_observation_df["Feature1"].sum(),
            places=5
        )
        self.assertAlmostEqual(
            data_to_fit_extra_features_data[:, 1].sum(),
            per_observation_df["Feature2"].sum(),
            places=5
        )
        self.assertAlmostEqual(
            data_to_fit_extra_features_data[:, 2].sum(),
            per_observation_df["Feature3"].sum(),
            places=5
        )
        self.assertAlmostEqual(
            data_to_fit_target_data.sum(),
            per_observation_df["Target"].sum()
        )
        self.assertEqual(2, per_channel_df.shape[0])
        self.assertAlmostEqual(data_to_fit.media_costs_scaled.sum(), per_channel_df["Cost"].sum())


if __name__ == '__main__':
    unittest.main()
