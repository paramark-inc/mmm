from datetime import date
import os
import tempfile
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

    def test_load_config_rejects_non_breaking_space_indentation(self):
        # Reproduces a real corrupted config: a nested block indented with
        # U+00A0 (non-breaking space) instead of regular spaces. YAML doesn't
        # treat NBSP as indentation, so it silently parses as sibling keys of
        # the media entry rather than a nested `priors:` block -- this should
        # raise loudly instead.
        contents = (
            "media:\n"
            "- display_name: Meta\n"
            "  priors: null\n"
            "    contribution_m: null\n"
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(contents)
            config_filename = f.name

        try:
            with self.assertRaises(ValueError) as ctx:
                load_config(config_filename)
            self.assertIn(f"{config_filename}:4", str(ctx.exception))
            self.assertIn("U+00A0", str(ctx.exception))
        finally:
            os.remove(config_filename)
