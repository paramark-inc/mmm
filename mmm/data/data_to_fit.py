import math
import numpy as np
import pandas as pd

from mmm.constants.constants import GRANULARITY_WEEKLY

from impl.lightweight_mmm.lightweight_mmm import preprocessing


class DataToFit:
    """
    InputData tranformed to be suitable for fitting a model.  The data undergoes the following transformations
    * split into train and test data set
    * scaled to be smaller values for better accuracy from the Bayesian model
    """

    @staticmethod
    def _robust_scaling_divide_operation(x):
        """
        Scaling divide operation that is robust to zero values (unlike jnp.mean).

        :param x: array of values
        :return: sum / count_of_positive_values
        """
        return x.sum() / (x > 0).sum()

    @staticmethod
    def from_input_data(input_data):
        """
        Generate a DataToFit instance from an InputData instance
        :param input_data: InputData instance
        :return: DataToFit instance
        """
        data_size = input_data.media_data.shape[0]

        split_point = math.ceil(data_size * 0.9)
        # split_point = math.ceil(data_size * 0.75)
        # split_point = math.ceil(data_size * 0.66)
        # split_point = math.ceil(data_size * 0.5)
        media_data_train = input_data.media_data[:split_point, :]
        media_data_test = input_data.media_data[split_point:, :]
        media_costs_by_row_train = input_data.media_costs_by_row[:split_point, :]
        media_costs_by_row_test = input_data.media_costs_by_row[split_point:, :]

        target_train = input_data.target_data[:split_point]
        target_test = input_data.target_data[split_point:]
        extra_features_train = input_data.extra_features_data[:split_point, :]
        extra_features_test = input_data.extra_features_data[split_point:, :]

        # Scale data (ignoring the zeroes in the media data).  Call fit_transform only the first time because only one
        # scaling constant is stored in the scaler.
        media_scaler = preprocessing.CustomScaler(
            divide_operation=DataToFit._robust_scaling_divide_operation
        )
        extra_features_scaler = preprocessing.CustomScaler(
            divide_operation=DataToFit._robust_scaling_divide_operation
        )
        target_scaler = preprocessing.CustomScaler(
            divide_operation=DataToFit._robust_scaling_divide_operation
        )

        # scale cost up by N since fit() will divide it by number of time periods
        media_cost_scaler = preprocessing.CustomScaler(
            divide_operation=DataToFit._robust_scaling_divide_operation
        )

        media_data_train_scaled = media_scaler.fit_transform(media_data_train)
        media_data_test_scaled = media_scaler.transform(media_data_test)
        extra_features_train_scaled = extra_features_scaler.fit_transform(extra_features_train)
        extra_features_test_scaled = extra_features_scaler.transform(extra_features_test)
        target_train_scaled = target_scaler.fit_transform(target_train)
        target_test_scaled = target_scaler.transform(target_test)

        media_priors_scaled = media_cost_scaler.fit_transform(input_data.media_priors)
        media_costs_scaled = media_cost_scaler.transform(input_data.media_costs)
        media_costs_by_row_train_scaled = media_cost_scaler.transform(media_costs_by_row_train)
        media_costs_by_row_test_scaled = media_cost_scaler.transform(media_costs_by_row_test)

        return DataToFit(
            date_strs=input_data.date_strs,
            time_granularity=input_data.time_granularity,
            media_data_train_scaled=media_data_train_scaled,
            media_data_test_scaled=media_data_test_scaled,
            media_scaler=media_scaler,
            media_costs_scaled=media_costs_scaled,
            media_priors_scaled=media_priors_scaled,
            media_costs_by_row_train_scaled=media_costs_by_row_train_scaled,
            media_costs_by_row_test_scaled=media_costs_by_row_test_scaled,
            media_costs_scaler=media_cost_scaler,
            media_names=input_data.media_names,
            extra_features_train_scaled=extra_features_train_scaled,
            extra_features_test_scaled=extra_features_test_scaled,
            extra_features_scaler=extra_features_scaler,
            extra_features_names=input_data.extra_features_names,
            target_train_scaled=target_train_scaled,
            target_test_scaled=target_test_scaled,
            target_is_log_scale=input_data.target_is_log_scale,
            target_scaler=target_scaler,
            target_name=input_data.target_name,
        )

    def __init__(
        self,
        date_strs,
        time_granularity,
        media_data_train_scaled,
        media_data_test_scaled,
        media_scaler,
        media_costs_scaled,
        media_priors_scaled,
        media_costs_by_row_train_scaled,
        media_costs_by_row_test_scaled,
        media_costs_scaler,
        media_names,
        extra_features_train_scaled,
        extra_features_test_scaled,
        extra_features_scaler,
        extra_features_names,
        target_train_scaled,
        target_test_scaled,
        target_is_log_scale,
        target_scaler,
        target_name,
    ):
        self.date_strs = date_strs
        self.time_granularity = time_granularity
        self.media_data_train_scaled = media_data_train_scaled
        self.media_data_test_scaled = media_data_test_scaled
        self.media_scaler = media_scaler
        self.media_costs_scaled = media_costs_scaled
        self.media_priors_scaled = media_priors_scaled
        self.media_costs_by_row_train_scaled = media_costs_by_row_train_scaled
        self.media_costs_by_row_test_scaled = media_costs_by_row_test_scaled
        self.media_costs_scaler = media_costs_scaler
        self.media_names = media_names
        self.extra_features_train_scaled = extra_features_train_scaled
        self.extra_features_test_scaled = extra_features_test_scaled
        self.extra_features_scaler = extra_features_scaler
        self.extra_features_names = extra_features_names
        self.target_train_scaled = target_train_scaled
        self.target_test_scaled = target_test_scaled
        self.target_is_log_scale = target_is_log_scale
        self.target_scaler = target_scaler
        self.target_name = target_name

    def to_data_frame(self, unscaled=False):
        """
        :param unscaled: True to unscale the data, False otherwise
        :return: (per-observation df, per-channel df) DataFrames to view DataToFit data.
                 All arrays are deep copies.
        """
        observation_data_by_column_name = {}

        media_data_scaled = np.vstack((self.media_data_train_scaled, self.media_data_test_scaled))
        media_data_unscaled = self.media_scaler.inverse_transform(media_data_scaled)
        n_media_channels = media_data_scaled.shape[1]
        for media_idx in range(n_media_channels):
            col_name = f"{self.media_names[media_idx]} volume"
            media_data_touse = media_data_unscaled if unscaled else media_data_scaled
            observation_data_by_column_name[col_name] = media_data_touse[:, media_idx]

        media_costs_by_row_scaled = np.vstack(
            (self.media_costs_by_row_train_scaled, self.media_costs_by_row_test_scaled)
        )
        media_costs_by_row_unscaled = self.media_costs_scaler.inverse_transform(
            media_costs_by_row_scaled
        )
        for media_idx in range(n_media_channels):
            col_name = f"{self.media_names[media_idx]} cost"
            media_costs_by_row_touse = (
                media_costs_by_row_unscaled if unscaled else media_costs_by_row_scaled
            )
            observation_data_by_column_name[col_name] = media_costs_by_row_touse[:, media_idx]

        extra_features_data_scaled = np.vstack(
            (self.extra_features_train_scaled, self.extra_features_test_scaled)
        )
        extra_features_data_unscaled = self.extra_features_scaler.inverse_transform(
            extra_features_data_scaled
        )
        for extra_features_idx in range(extra_features_data_scaled.shape[1]):
            col_name = self.extra_features_names[extra_features_idx]
            extra_features_data_touse = (
                extra_features_data_unscaled if unscaled else extra_features_data_scaled
            )
            observation_data_by_column_name[col_name] = extra_features_data_touse[
                :, extra_features_idx
            ]

        target_data_scaled = np.hstack((self.target_train_scaled, self.target_test_scaled))
        target_data_unscaled = self.target_scaler.inverse_transform(target_data_scaled)
        target_data_touse = target_data_unscaled if unscaled else target_data_scaled
        observation_data_by_column_name[self.target_name] = target_data_touse

        # TODO push conversion to datetime upstream so that it is common across all data sets
        per_observation_df = pd.DataFrame(
            data=observation_data_by_column_name,
            index=pd.DatetimeIndex(
                pd.to_datetime(self.date_strs, dayfirst=False, yearfirst=False),
                freq="W" if self.time_granularity == GRANULARITY_WEEKLY else "D",
            ),
            dtype=np.float64,
            copy=True,
        )

        channel_data_by_column_name = {}

        media_costs_unscaled = self.media_costs_scaler.inverse_transform(self.media_costs_scaled)
        media_costs_touse = media_costs_unscaled if unscaled else self.media_costs_scaled
        channel_data_by_column_name["Cost"] = media_costs_touse

        media_priors_unscaled = self.media_costs_scaler.inverse_transform(self.media_priors_scaled)
        media_priors_touse = media_priors_unscaled if unscaled else self.media_priors_scaled
        channel_data_by_column_name["Prior"] = media_priors_touse

        per_channel_df = pd.DataFrame(
            data=channel_data_by_column_name, index=self.media_names, dtype=np.float64, copy=True
        )

        return per_observation_df, per_channel_df
