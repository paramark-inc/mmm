import numpy as np
import os
import pandas as pd

import unittest
from numpy.testing import assert_array_equal, assert_array_almost_equal
from pandas.testing import assert_series_equal

from base_driver.config import load_config
from mmm.constants import constants
from mmm.parser.csv import parse_csv_generic
from mmm.transform.transform import transform_input_generic

from .test_input_data import InputDataTest
from mmm.data import DataToFit, InputData


class DataToFitTest(unittest.TestCase):
    config_filename = os.path.join(os.path.dirname(__file__), "..", "test.yaml")
    csv_filename = os.path.join(os.path.dirname(__file__), "..", "test.csv")

    # if you add or modify attributes on DataToFit and need to re-generate this file, try doing:
    # 1. Modify test_data_to_fit_to_dict to dump the data_to_fit to a gzip file by adding
    #      'data_to_fit.dump(".")'.  This will create a file called data_to_fit.gz in
    #      the directory where you run the test.
    # 2. Replace test/test_data_to_fit.gz with this file.
    gz_filename = os.path.join(os.path.dirname(__file__), "..", "test_data_to_fit.gz")

    # noinspection PyMethodMayBeStatic
    def test_data_to_fit_from_input_data(self):
        input_data = InputDataTest.generate_test_input_data()
        data_to_fit = DataToFit.from_input_data(input_data=input_data, config={})
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

    def test_data_to_fit_with_train_only(self):
        input_data = InputDataTest.generate_test_input_data()

        data_to_fit = DataToFit.from_input_data(
            input_data=input_data, config={"train_test_ratio": 1.0}
        )

    def test_data_to_fit_to_dict(self):
        _, config = load_config(self.config_filename)
        data_dict = parse_csv_generic(self.csv_filename, config)
        input_data = transform_input_generic(data_dict, config)

        data_to_fit = DataToFit.from_input_data(input_data=input_data, config=config)

        output = data_to_fit.to_dict()

        assert_array_almost_equal(
            output["data"]["media_cost_priors_scaled"], [1.5714285, 0.42857143]
        )
        assert_array_equal(
            output["display"]["date_strs"], [f"2023-01-{n:02}" for n in range(1, 11)]
        )

        self.assertEqual(output["display"]["media_names"], ["Channel 1", "Channel 2"])
        self.assertEqual(output["display"]["extra_features_names"], [])

        self.assertDictEqual(
            output["scalers"]["target_scaler"], {"multiply_by": 1.0, "divide_by": 55.0}
        )

        # check to make sure we haven't forgotten any attributes in to_dict();
        # assumes that all the keys of the dict (one level down) are also all
        # the properties of the object (which we count via a hack on __dict__)
        dict_count = 0
        for section in output.values():
            dict_count += len(section)
        self.assertEqual(len(data_to_fit.__dict__), dict_count)

    def test_data_to_fit_from_gz_file(self):
        data_to_fit = DataToFit.from_file(self.gz_filename)

        self.assertIsInstance(data_to_fit, DataToFit)

        assert_array_almost_equal(
            data_to_fit.target_train_scaled,
            [
                0.18181819,
                0.36363637,
                0.54545456,
                0.72727275,
                0.90909094,
                1.0909091,
                1.2727273,
                1.4545455,
                1.6363636,
            ],
        )

        assert_array_almost_equal(
            data_to_fit.target_test_scaled,
            [1.8181819],
        )

    def test_data_to_fit_to_data_frame_from_input_data(self):
        input_data = InputDataTest.generate_test_input_data()

        data_to_fit = DataToFit.from_input_data(input_data=input_data, config={})
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

    def test_data_to_fit_to_data_frame_from_gz(self):
        data_to_fit = DataToFit.from_file(self.gz_filename)

        # omit last row of CSV because it's excluded in parser by date range
        csv_df = pd.read_csv(self.csv_filename)[:-1]

        per_observation_df, per_channel_df = data_to_fit.to_data_frame(unscaled=True)

        self.assertEqual(csv_df["Spend1"].sum(), per_channel_df["Cost"].loc["Channel 1"])

        assert_series_equal(
            pd.to_datetime(csv_df["Date"]),
            per_observation_df.index.to_series(),
            check_index=False,
            check_names=False,
        )

        assert_series_equal(
            csv_df["Impressions1"].astype(float),
            per_observation_df["Channel 1 volume"],
            check_index=False,
            check_names=False,
        )
        assert_series_equal(
            csv_df["Spend2"].astype(float),
            per_observation_df["Channel 2 cost"],
            check_index=False,
            check_names=False,
        )
        assert_series_equal(
            csv_df["Target"].astype(float),
            per_observation_df["Target"],
            check_index=False,
            check_names=False,
        )
