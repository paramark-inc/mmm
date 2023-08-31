import numpy as np
import os
import unittest

from base_driver.config import load_config
from mmm.parser.csv import parse_csv_generic


class CSVParserTest(unittest.TestCase):
    def test_parse_csv_generic(self):
        config_filename = os.path.join(os.path.dirname(__file__), "..", "test.yaml")
        input_filename = os.path.join(os.path.dirname(__file__), "..", "test.csv")

        _, config = load_config(config_filename)
        data_dict = parse_csv_generic(input_filename, config)

        self.assertEqual(data_dict["granularity"], config["raw_data_granularity"])

        np.testing.assert_array_equal(
            data_dict["date_strs"], [f"2023-01-{n:02}" for n in range(1, 11)]
        )

        np.testing.assert_array_equal(data_dict["metrics"]["Spend1"], range(1, 11))

        np.testing.assert_array_equal(data_dict["metrics"]["Impressions2"], [0.0, 1.0] * 5)

        np.testing.assert_array_equal(data_dict["metrics"]["Target"], range(10, 110, 10))
