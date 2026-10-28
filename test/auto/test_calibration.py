from datetime import date, timedelta
import os
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

from mmm.constants import constants
from mmm.data import DataToFit, InputData
from mmm.describe.describe import get_calibration_check_df
from mmm.fit.calibration import build_calibration, to_model_calibration
from mmm.fit.fit import fit_lightweight_mmm


def _weekly_input_data(n_weeks=104, seed=0):
    """
    Two collinear channels where only channel A drives the target, so an uncalibrated model
    can't tell them apart well.  Channel A spends nothing in the first 4 weeks.
    """
    rng = np.random.default_rng(seed)
    date_strs = [
        (date.fromisoformat("2024-01-07") + timedelta(weeks=idx)).isoformat()
        for idx in range(n_weeks)
    ]
    spend_a = 1000.0 + 400.0 * np.sin(np.arange(n_weeks) / 6.0) + rng.normal(0, 50, n_weeks)
    spend_a[:4] = 0.0
    spend_b = 0.8 * spend_a + rng.normal(0, 30, n_weeks)
    spend_b[:4] = 0.0
    media_costs_by_row = np.stack([spend_a, spend_b], axis=1).clip(min=0.0)
    media_costs = media_costs_by_row.sum(axis=0)
    target = 500.0 + 0.5 * media_costs_by_row[:, 0] + rng.normal(0, 20, n_weeks)

    return InputData(
        date_strs=np.array(date_strs),
        time_granularity=constants.GRANULARITY_WEEKLY,
        media_data=media_costs_by_row.copy(),
        media_costs=media_costs,
        media_costs_by_row=media_costs_by_row,
        media_cost_priors=media_costs,
        learned_media_priors=np.zeros(shape=media_costs.shape),
        media_names=["A", "B"],
        extra_features_data=rng.normal(10, 1, (n_weeks, 1)),
        extra_features_names=["Noise"],
        target_data=target,
        target_is_log_scale=False,
        target_name="Target",
        geo_names=None,
    )


def _entry(**overrides):
    entry = {
        "channels": ["A"],
        "start_date": "2024-03-10",
        "end_date": "2024-04-20",
        "cpic_low": 1.5,
        "cpic_high": 2.5,
    }
    entry.update(overrides)
    return entry


class BuildCalibrationTest(unittest.TestCase):
    def setUp(self):
        self.data_to_fit = DataToFit.from_input_data(
            input_data=_weekly_input_data(), config={"train_test_ratio": 1.0}
        )

    def test_no_calibration(self):
        self.assertEqual(build_calibration({}, self.data_to_fit), [])
        self.assertEqual(build_calibration({"calibration": []}, self.data_to_fit), [])

    def test_window_overlap_and_lift(self):
        # weeks start 2024-01-07, 01-14, 01-21, ...; the week of 01-07 ends 01-13
        entries = build_calibration(
            {"calibration": [_entry(start_date="2024-02-07", end_date="2024-02-18")]},
            self.data_to_fit,
        )
        self.assertEqual(len(entries), 1)
        entry = entries[0]
        # rows starting 2024-02-04 and 2024-02-11 overlap; 2024-02-18 also starts in the window
        assert_array_equal(np.nonzero(entry.row_mask)[0], [4, 5, 6])
        assert_array_equal(entry.channel_mask, [1.0, 0.0])

        expected_spend = self.data_to_fit.media_costs_scaler.inverse_transform(
            self.data_to_fit.media_costs_by_row_train_scaled
        )[4:7, 0].sum()
        self.assertAlmostEqual(entry.window_spend, float(expected_spend), places=0)

        lift_low, lift_high = entry.window_spend / 2.5, entry.window_spend / 1.5
        self.assertAlmostEqual(entry.lift_mean, (lift_low + lift_high) / 2)
        # default interval is 0.9, z = 1.6449
        self.assertAlmostEqual(entry.lift_sd, (lift_high - lift_low) / (2 * 1.6448536), places=3)

    def test_interval_and_multiple_channels(self):
        entries = build_calibration(
            {"calibration": [_entry(channels=["A", "B"], interval=0.95)]}, self.data_to_fit
        )
        entry = entries[0]
        assert_array_equal(entry.channel_mask, [1.0, 1.0])
        lift_low, lift_high = entry.window_spend / 2.5, entry.window_spend / 1.5
        self.assertAlmostEqual(entry.lift_sd, (lift_high - lift_low) / (2 * 1.959964), places=3)

    def test_to_model_calibration_scales_lift(self):
        entries = build_calibration(
            {"calibration": [_entry(), _entry(channels=["B"])]}, self.data_to_fit
        )
        model_calibration = to_model_calibration(entries, self.data_to_fit)
        self.assertEqual(model_calibration["row_mask"].shape, (2, 104))
        self.assertEqual(model_calibration["channel_mask"].shape, (2, 2))
        divide_by = float(self.data_to_fit.target_scaler.divide_by)
        assert_array_almost_equal(
            model_calibration["lift_mean"],
            np.array([e.lift_mean for e in entries]) / divide_by,
            decimal=5,
        )
        assert_array_almost_equal(
            model_calibration["lift_sd"],
            np.array([e.lift_sd for e in entries]) / divide_by,
            decimal=5,
        )

    def test_errors(self):
        cases = {
            "unknown channels": _entry(channels=["C"]),
            "non-empty list": _entry(channels="A"),
            "no rows": _entry(start_date="2030-01-01", end_date="2030-02-01"),
            "no spend": _entry(start_date="2024-01-07", end_date="2024-01-20"),
            "end_date is before": _entry(start_date="2024-04-20", end_date="2024-03-10"),
            "cpic_low < cpic_high": _entry(cpic_low=3.0, cpic_high=2.0),
            "interval": _entry(interval=1.0),
            "not a date": _entry(start_date="soon"),
        }
        for message, entry in cases.items():
            with self.subTest(message):
                with self.assertRaisesRegex(ValueError, message):
                    build_calibration({"calibration": [entry]}, self.data_to_fit)

    def test_rejects_log_target_and_geo_models(self):
        config = {"calibration": [_entry()]}

        self.data_to_fit.target_is_log_scale = True
        with self.assertRaisesRegex(ValueError, "log_scale_target"):
            build_calibration(config, self.data_to_fit)

        self.data_to_fit.target_is_log_scale = False
        self.data_to_fit.geo_names = ["US"]
        with self.assertRaisesRegex(ValueError, "national"):
            build_calibration(config, self.data_to_fit)


class CalibratedFitTest(unittest.TestCase):
    """Fits real (small) models, so this is the slowest test in this file."""

    config = {
        "model_name": "adstock",
        "degrees_seasonality": 1,
        "number_warmup": 400,
        "number_samples": 400,
        "number_chains": 1,
        "seed": 1,
    }

    def _fit(self, config):
        data_to_fit = DataToFit.from_input_data(
            input_data=_weekly_input_data(), config={"train_test_ratio": 1.0}
        )
        with tempfile.TemporaryDirectory() as tmpdirname:
            model = fit_lightweight_mmm(
                config=config, data_to_fit=data_to_fit, results_dir=tmpdirname
            )
            with open(os.path.join(tmpdirname, "fit_params.yaml")) as f:
                fit_params_yaml = f.read()
        return model, data_to_fit, fit_params_yaml

    def test_calibration_pulls_cpic_into_range(self):
        # Channel A's true CPIC is 2.0; ask for a clearly different, tight range so the test
        # checks that the term moves the posterior rather than that the data already agrees.
        calibrated_config = {
            **self.config,
            "calibration": [_entry(cpic_low=3.6, cpic_high=4.4, interval=0.9)],
        }

        uncalibrated, data_to_fit, fit_params_yaml = self._fit(self.config)
        self.assertNotIn("calibration_model_lift", uncalibrated.trace)
        self.assertNotIn("calibration", fit_params_yaml)
        self.assertIsNone(get_calibration_check_df(uncalibrated, data_to_fit, self.config))

        calibrated, data_to_fit, fit_params_yaml = self._fit(calibrated_config)
        self.assertEqual(calibrated.trace["calibration_model_lift"].shape, (400, 1))
        self.assertIn("calibration:", fit_params_yaml)
        self.assertIn("lift_mean", fit_params_yaml)

        check = get_calibration_check_df(calibrated, data_to_fit, calibrated_config)
        self.assertEqual(len(check), 1)
        self.assertEqual(check.loc[0, "channels"], "A")
        model_cpic = check.loc[0, "model_cpic_median"]
        self.assertGreater(model_cpic, 3.0)
        self.assertLess(model_cpic, 5.0)

        # Predictions don't depend on calibration; they use the same posterior machinery.
        predictions = calibrated.predict(
            media=data_to_fit.media_data_train_scaled[-4:],
            extra_features=data_to_fit.extra_features_train_scaled[-4:],
            seed=1,
        )
        self.assertEqual(predictions.shape, (400, 4))


if __name__ == "__main__":
    unittest.main()
