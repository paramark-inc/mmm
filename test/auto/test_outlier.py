import numpy as np
from numpy.testing import assert_array_almost_equal
import unittest

from mmm.constants import constants
from mmm.data import InputData
from mmm.outlier.outlier import remove_outliers_from_input


class OutlierTestCase(unittest.TestCase):
    # noinspection PyMethodMayBeStatic
    def test_remove_outliers_from_input_trimmed_mean(self):
        input_data = InputData(
            date_strs=np.array(["1/1", "1/2", "1/3", "1/4"]),
            time_granularity=constants.GRANULARITY_DAILY,
            media_data=np.array(
                [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0], [4.0, 4.0, 4.0]],
                dtype=np.float64,
            ),
            media_costs=np.array([100.0, 200.0, 300.0], dtype=np.float64),
            media_costs_by_row=np.ndarray(shape=(0, 3)),
            media_cost_priors=np.array([100.0, 200.0, 300.0], dtype=np.float64),
            learned_media_priors=np.zeros(shape=(3,)),
            media_names=["Google", "Facebook", "Events"],
            extra_features_data=np.array(
                [[5.0, 5.0], [6.0, 6.0], [7.0, 7.0], [8.0, 8.0]], dtype=np.float64
            ),
            extra_features_names=["Macro1", "Macro2"],
            target_data=np.array([9.0, 10.0, 11.0, 12.0], dtype=np.float64),
            target_is_log_scale=False,
            target_name="Sales",
        )

        media_data_outliers = {"Google": [0, 3]}
        extra_features_outliers = {}
        target_outliers = [1]

        input_data_outliers_removed = remove_outliers_from_input(
            input_data=input_data,
            media_data_outliers=media_data_outliers,
            extra_features_outliers=extra_features_outliers,
            target_outliers=target_outliers,
            removal_type=constants.REMOVE_OUTLIERS_TYPE_REPLACE_WITH_TRIMMED_MEAN,
        )

        expected_media_data = np.array(
            [[2.5, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0], [2.5, 4.0, 4.0]], dtype=np.float64
        )
        assert_array_almost_equal(expected_media_data, input_data_outliers_removed.media_data)
        assert_array_almost_equal(
            input_data.extra_features_data, input_data_outliers_removed.extra_features_data
        )

        expected_target_data = np.array([9.0, 10.5, 11.0, 12.0], dtype=np.float64)
        assert_array_almost_equal(expected_target_data, input_data_outliers_removed.target_data)

    # noinspection PyMethodMayBeStatic
    def test_remove_outliers_from_input_p10(self):
        input_data = InputData(
            date_strs=np.array(["1/1", "1/2", "1/3", "1/4"]),
            time_granularity=constants.GRANULARITY_DAILY,
            media_data=np.array(
                [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0], [4.0, 4.0, 4.0]],
                dtype=np.float64,
            ),
            media_costs=np.array([100.0, 200.0, 300.0], dtype=np.float64),
            media_costs_by_row=np.ndarray(shape=(0, 3)),
            media_cost_priors=np.array([100.0, 200.0, 300.0], dtype=np.float64),
            learned_media_priors=np.zeros(shape=(3,)),
            media_names=["Google", "Facebook", "Events"],
            extra_features_data=np.array(
                [[5.0, 5.0], [6.0, 6.0], [7.0, 7.0], [8.0, 8.0]], dtype=np.float64
            ),
            extra_features_names=["Macro1", "Macro2"],
            target_data=np.array([9.0, 10.0, 11.0, 12.0], dtype=np.float64),
            target_is_log_scale=False,
            target_name="Sales",
        )

        media_data_outliers = {"Google": [0, 3]}
        extra_features_outliers = {}
        target_outliers = [1]

        input_data_outliers_removed = remove_outliers_from_input(
            input_data=input_data,
            media_data_outliers=media_data_outliers,
            extra_features_outliers=extra_features_outliers,
            target_outliers=target_outliers,
            removal_type=constants.REMOVE_OUTLIERS_TYPE_REPLACE_WITH_P10_VALUE,
        )

        expected_media_data = np.array(
            [[1.3, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0], [1.3, 4.0, 4.0]], dtype=np.float64
        )
        assert_array_almost_equal(expected_media_data, input_data_outliers_removed.media_data)
        assert_array_almost_equal(
            input_data.extra_features_data, input_data_outliers_removed.extra_features_data
        )

        expected_target_data = np.array([9.0, 9.3, 11.0, 12.0], dtype=np.float64)
        assert_array_almost_equal(expected_target_data, input_data_outliers_removed.target_data)


if __name__ == "__main__":
    unittest.main()
