import numpy as np
import os
import tempfile
import unittest

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
                config=config,
                data_to_fit=data_to_fit,
                results_dir=tmpdirname,
            )

            # first verify model params
            self.assertEqual(model.custom_priors, {})
            self.assertEqual(model.media_names, ["Channel 1", "Channel 2"])
            self.assertEqual(model.n_geos, 1)
            self.assertEqual(model.n_media_channels, 2)
            self.assertListEqual(os.listdir(tmpdirname), ["fit_params.yaml"])

            # then verify model fit by looking at shapes of objects in the trace
            # (we can't look at values directly because even with a fixed seed,
            # JAX's RNG behaves differently on different CPU architectures)
            self.assertEqual(model.trace["ad_effect_retention_rate"].shape, (10, 2))
            self.assertEqual(model.trace["coef_media"].shape, (10, 2))
            self.assertEqual(model.trace["coef_trend"].shape, (10, 1))
            self.assertEqual(model.trace["expo_trend"].shape, (10,))
            self.assertEqual(model.trace["exponent"].shape, (10, 2))
            self.assertEqual(model.trace["gamma_seasonality"].shape, (10, 1, 2))
            self.assertEqual(model.trace["intercept"].shape, (10, 1))
            self.assertEqual(model.trace["media_transformed"].shape, (10, 9, 2))
            self.assertEqual(model.trace["mu"].shape, (10, 9))
            self.assertEqual(model.trace["peak_effect_delay"].shape, (10, 2))
            self.assertEqual(model.trace["sigma"].shape, (10, 1))
            self.assertEqual(model.trace["weekday"].shape, (10, 7))

            # finally, check that the model makes predictions as expected, with
            # another shape check, and a loose check on the output values
            prediction_input = np.array([[11, 0], [12, 1], [13, 0], [14, 1]], np.float32)

            prediction = model.predict(
                media=data_to_fit.media_scaler.transform(prediction_input),
                extra_features=None,
                target_scaler=data_to_fit.target_scaler,
            )

            self.assertEqual(prediction.shape, (10, 4))

            mean_predictions = prediction.mean(axis=1)
            assert np.all(
                mean_predictions > 50
            ), f"expected prediction means greater than 50, got: {mean_predictions}"
            assert np.all(
                mean_predictions < 200
            ), f"expected prediction means less than 200, got: {mean_predictions}"
