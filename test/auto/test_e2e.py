import numpy as np
import os
import platform
import tempfile
import unittest
from numpy.testing import assert_array_almost_equal

from base_driver.config import load_config
from mmm.data import DataToFit
from mmm.fit.fit import fit_lightweight_mmm


class EndToEndTest(unittest.TestCase):
    config_filename = os.path.join(os.path.dirname(__file__), "..", "test.yaml")

    # This test will fail if attributes in the DataToFit object don't match up with
    # the serialized contents of this file (see notes in test_data_to_fit.py).
    gz_filename = os.path.join(os.path.dirname(__file__), "..", "test_data_to_fit.gz")

    def test_mmm_fit_predict(self):
        _, config = load_config(self.config_filename)
        data_to_fit = DataToFit.from_file(self.gz_filename)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model = fit_lightweight_mmm(
                data_to_fit=data_to_fit,
                model_name=config["model_name"],
                results_dir=tmpdirname,
                degrees_seasonality=config["degrees_seasonality"],
                weekday_seasonality=config["weekday_seasonality"],
                number_warmup=config["number_warmup"],
                number_samples=config["number_samples"],
                number_chains=config["number_chains"],
                seed=config.get("seed"),
                custom_prior_config=config.get("custom_priors"),
            )

            # first verify model params
            self.assertEqual(model.custom_priors, {})
            self.assertEqual(model.media_names, ["Channel 1", "Channel 2"])
            self.assertEqual(model.n_geos, 1)
            self.assertEqual(model.n_media_channels, 2)
            self.assertListEqual(os.listdir(tmpdirname), ["fit_params.yaml"])

            # then verify model fit by looking at random values from the trace
            # note that even with a fixed seed, JAX's random number generator
            # produces (deterministically) different results on different CPUs
            if platform.machine() == "x86_64":
                expected = {
                    "coef_media": [1.6870707, 0.17920175],
                    "expo_trend": 1.3377011,
                    "gamma_seasonality": [[-0.885461, 0.07778881]],
                    "intercept": 0.08363592,
                    "peak_effect_delay": [0.23231119, 1.6421096],
                    "prediction": [117.0781174, 108.2175751, 110.4735489, 114.1343918],
                }
            elif platform.machine() == "arm64":
                expected = {
                    "coef_media": [0.9149572, 0.12196853],
                    "expo_trend": 1.2901471,
                    "gamma_seasonality": [[-8.8577402e-01, -2.1908896e-03]],
                    "intercept": 2.1729574,
                    "peak_effect_delay": [0.05093075, 1.7051804],
                    "prediction": [118.0228882, 125.4357605, 123.3989105, 150.4321747],
                }

            assert_array_almost_equal(model.trace["coef_media"][8], expected["coef_media"])
            self.assertAlmostEqual(
                model.trace["expo_trend"][9],
                expected["expo_trend"],
            )
            assert_array_almost_equal(
                model.trace["gamma_seasonality"][0],
                expected["gamma_seasonality"],
            )
            self.assertAlmostEqual(
                model.trace["intercept"][2][0],
                expected["intercept"],
            )
            assert_array_almost_equal(
                model.trace["peak_effect_delay"][1],
                expected["peak_effect_delay"],
            )

            # finally, verify that the model makes predictions as expected
            prediction_input = np.array([[11, 0], [12, 1], [13, 0], [14, 1]], np.float32)

            prediction = model.predict(
                media=data_to_fit.media_scaler.transform(prediction_input),
                extra_features=None,
                target_scaler=data_to_fit.target_scaler,
            )

            assert_array_almost_equal(prediction[9], expected["prediction"])
