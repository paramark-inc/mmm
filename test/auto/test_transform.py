import numpy as np
import os
import unittest

from base_driver.config import load_config
from mmm.parser.csv import parse_csv_generic
from mmm.transform.transform import transform_input_generic

from mmm.data import InputData


class TransformTest(unittest.TestCase):
    def test_transform_input_generic(self):
        config_filename = os.path.join(os.path.dirname(__file__), "..", "test.yaml")
        input_filename = os.path.join(os.path.dirname(__file__), "..", "test.csv")

        _, config = load_config(config_filename)
        data_dict = parse_csv_generic(input_filename, config)
        input_data = transform_input_generic(data_dict, config)

        self.assertIsInstance(input_data, InputData)

        self.assertEqual(input_data.target_is_log_scale, config["log_scale_target"])
        self.assertEqual(input_data.target_name, config["target_col"])
        self.assertEqual(input_data.media_names, [x["display_name"] for x in config["media"]])
        self.assertEqual(input_data.extra_features_names, [])

        np.testing.assert_array_equal(input_data.media_costs, [55, 15])
        np.testing.assert_array_equal(input_data.media_cost_priors, [55, 15])

        # verify the two sets of impressions data (columns of input_data.media_data)
        np.testing.assert_array_equal(input_data.media_data[:, 0], range(1, 11))
        np.testing.assert_array_equal(input_data.media_data[:, 1], [0, 1] * 5)

        np.testing.assert_array_equal(input_data.target_data, range(10, 110, 10))
