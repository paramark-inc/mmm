import numpy as np
import os
import unittest

from base_driver.config import load_config
from mmm.parser.csv import parse_csv_generic
from mmm.transform.transform import transform_input_generic

from mmm.data import DataToFit


class DataToFitTest(unittest.TestCase):
    def test_data_to_fit_to_dict(self):
        config_filename = os.path.join(os.path.dirname(__file__), "..", "test.yaml")
        input_filename = os.path.join(os.path.dirname(__file__), "..", "test.csv")

        _, config = load_config(config_filename)
        data_dict = parse_csv_generic(input_filename, config)
        input_data = transform_input_generic(data_dict, config)

        data_to_fit = DataToFit.from_input_data(input_data=input_data)

        output = data_to_fit.to_dict()

        np.testing.assert_array_almost_equal(
            output["data"]["media_priors_scaled"], [1.5714285, 0.42857143]
        )
        np.testing.assert_array_equal(
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
        filename = os.path.join(os.path.dirname(__file__), "..", "test_data_to_fit.gz")
        data_to_fit = DataToFit.from_file(filename)

        self.assertIsInstance(data_to_fit, DataToFit)

        np.testing.assert_array_almost_equal(
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

        np.testing.assert_array_almost_equal(
            data_to_fit.target_test_scaled,
            [1.8181819],
        )
