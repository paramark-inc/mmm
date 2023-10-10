import numpy as np
import os
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
            assert_array_almost_equal(
                model.trace["ad_effect_retention_rate"][4], [0.8497401, 0.6917644]
            )
            assert_array_almost_equal(model.trace["coef_media"][8], [0.9149572, 0.12196853])
            self.assertAlmostEqual(model.trace["coef_trend"][1][0], 0.04331071)
            self.assertAlmostEqual(
                model.trace["expo_trend"][9],
                1.2901471,
            )
            assert_array_almost_equal(
                model.trace["exponent"][3],
                [0.95118594, 0.9256289],
            )
            assert_array_almost_equal(
                model.trace["gamma_seasonality"][0],
                [[-8.8577402e-01, -2.1908896e-03]],
            )
            self.assertAlmostEqual(model.trace["intercept"][2][0], 2.1729574)
            assert_array_almost_equal(
                model.trace["media_transformed"][5][2],
                [3.0492800e-01, 4.8702326e-01],
            )
            self.assertAlmostEqual(
                model.trace["mu"][2][2],
                0.6117116,
            )
            assert_array_almost_equal(
                model.trace["peak_effect_delay"][1],
                [0.05093075, 1.7051804],
            )
            self.assertAlmostEqual(model.trace["sigma"][6][0], 0.24445353)
            self.assertAlmostEqual(
                model.trace["weekday"][7][3],
                0.3071486,
            )

            # finally, verify that the model makes predictions as expected
            prediction_input = np.array([[11, 0], [12, 1], [13, 0], [14, 1]], np.float32)

            prediction = model.predict(
                media=data_to_fit.media_scaler.transform(prediction_input),
                extra_features=None,
                target_scaler=data_to_fit.target_scaler,
            )

            assert_array_almost_equal(
                prediction[9], [118.0228882, 125.4357605, 123.3989105, 150.4321747]
            )
