from datetime import date
import os
import unittest

from base_driver.config import load_config


class ConfigTest(unittest.TestCase):
    def test_load_config(self):
        config_filename = os.path.join(os.path.dirname(__file__), "..", "test.yaml")

        raw_config, config = load_config(config_filename)

        self.assertIn("\nraw_data_granularity: daily\n", raw_config)

        self.assertEqual(config["seed"], 1)

        self.assertDictEqual(
            config["data_rows"],
            {
                "start_date": date(year=2023, month=1, day=1),
                "end_date": date(year=2023, month=1, day=10),
            },
        )
