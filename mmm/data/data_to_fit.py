import gzip
import math
import msgpack
import msgpack_numpy
import numpy as np
import os
import pandas as pd

from mmm.constants import constants
from mmm.data.input_data import InputData
from mmm.data.serializable_scaler import SerializableScaler

msgpack_numpy.patch()


class DataToFit:
    """
    InputData tranformed to be suitable for fitting a model.  The data undergoes the following transformations
    * split into train and test data set
    * scaled to be smaller values for better accuracy from the Bayesian model
    """

    @staticmethod
    def from_input_data(input_data: InputData, config: dict):
        """
        Generate a DataToFit instance from an InputData instance
        :param input_data: InputData instance
        :return: DataToFit instance
        """
        data_size = input_data.media_data.shape[0]

        split_ratio = config.get("train_test_ratio", 0.9)
        split_point = math.ceil(data_size * split_ratio)

        media_data_train = input_data.media_data[:split_point, :]
        media_costs_by_row_train = input_data.media_costs_by_row[:split_point, :]
        extra_features_train = input_data.extra_features_data[:split_point, :]
        target_train = input_data.target_data[:split_point]

        n_media_channels = len(input_data.media_names)
        n_extra_features = len(input_data.extra_features_names)

        if split_point < data_size:
            has_test_dataset = True
            media_data_test = input_data.media_data[split_point:, :]
            media_costs_by_row_test = input_data.media_costs_by_row[split_point:, :]
            extra_features_test = input_data.extra_features_data[split_point:, :]
            target_test = input_data.target_data[split_point:]
        else:
            has_test_dataset = False
            media_data_test = np.ndarray(shape=(0, n_media_channels))
            media_costs_by_row_test = np.ndarray(shape=(0, n_media_channels))
            extra_features_test = np.ndarray(shape=(0, n_extra_features))
            target_test = np.array([])

        # Scale data (ignoring the zeroes in the media data).  Call fit only the first time because
        # only one scaling constant is stored in the scaler.
        media_scaler = SerializableScaler()
        extra_features_scaler = SerializableScaler()
        target_scaler = SerializableScaler()
        media_cost_scaler = SerializableScaler()

        # we fit based on the full data set to get a scaler that will work well for both train and
        # test rather than working well for only one or the other.  Consider the case where the
        # values in the train set are all zero, but the values in the test set are very large.  If
        # we fit based on the train we would generate a divide_by value of 1 which would not work
        # well for the data in the test set.
        media_scaler.fit(input_data.media_data)
        extra_features_scaler.fit(input_data.extra_features_data)
        target_scaler.fit(input_data.target_data)

        # note that we use media_cost_priors to fit the scaler, and then apply the scaling to
        # media_costs and media_costs_by_row in addition to media_cost_priors.  Ordinarily it
        # would be better to have a separate scaler for each, but since only media_cost_priors
        # is used for fitting the MMM model, this achieves the same result.
        media_cost_scaler.fit(input_data.media_cost_priors)

        media_data_train_scaled = media_scaler.transform(media_data_train)
        extra_features_train_scaled = extra_features_scaler.transform(extra_features_train)
        target_train_scaled = target_scaler.transform(target_train)

        if has_test_dataset:
            media_data_test_scaled = media_scaler.transform(media_data_test)
            extra_features_test_scaled = extra_features_scaler.transform(extra_features_test)
            target_test_scaled = target_scaler.transform(target_test)
        else:
            media_data_test_scaled = np.ndarray(shape=media_data_test.shape)
            extra_features_test_scaled = np.ndarray(shape=extra_features_test.shape)
            target_test_scaled = np.array([])

        media_cost_priors_scaled = media_cost_scaler.transform(input_data.media_cost_priors)
        media_costs_scaled = media_cost_scaler.transform(input_data.media_costs)
        media_costs_by_row_train_scaled = media_cost_scaler.transform(media_costs_by_row_train)
        media_costs_by_row_test_scaled = media_cost_scaler.transform(media_costs_by_row_test)

        return DataToFit(
            date_strs=input_data.date_strs,
            time_granularity=input_data.time_granularity,
            has_test_dataset=has_test_dataset,
            media_data_train_scaled=media_data_train_scaled,
            media_data_test_scaled=media_data_test_scaled,
            media_scaler=media_scaler,
            media_costs_scaled=media_costs_scaled,
            media_cost_priors_scaled=media_cost_priors_scaled,
            learned_media_priors=input_data.learned_media_priors.copy(),
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

    @staticmethod
    def from_dict(input_dict):
        """
        Recreate a DataToFit object from a dictionary created by to_dict().

        Scaler objects have to be manually recreated; other dict keys can
        be passed directly to the constructor.
        """
        scalers = {}
        for name, scaler_dict in input_dict["scalers"].items():
            scaler = SerializableScaler()
            scaler.from_dict(scaler_dict)
            scalers[name] = scaler

        return DataToFit(
            **scalers,
            **input_dict["data"],
            **input_dict["display"],
            **input_dict["config"],
        )

    @staticmethod
    def from_file(input_file):
        if input_file.endswith(".gz"):
            with gzip.open(input_file) as f:
                input_dict = msgpack.load(f)
        else:
            with open(input_file) as f:
                input_dict = msgpack.load(f)
        return DataToFit.from_dict(input_dict)

    def __init__(
        self,
        date_strs,
        time_granularity,
        has_test_dataset,
        media_data_train_scaled,
        media_data_test_scaled,
        media_scaler,
        media_costs_scaled,
        media_cost_priors_scaled,
        learned_media_priors,
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
        self.has_test_dataset = has_test_dataset
        self.media_data_train_scaled = media_data_train_scaled
        self.media_data_test_scaled = media_data_test_scaled
        self.media_scaler = media_scaler
        self.media_costs_scaled = media_costs_scaled
        self.media_cost_priors_scaled = media_cost_priors_scaled
        self.learned_media_priors = learned_media_priors
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

    def to_dict(self):
        return {
            "data": {
                # each of these attributes are JAX arrays, so convert them
                # to standard numpy before outputting (for msgpack_numpy)
                name: np.array(getattr(self, name))
                for name in [
                    "media_data_train_scaled",
                    "media_data_test_scaled",
                    "media_costs_scaled",
                    "media_cost_priors_scaled",
                    "learned_media_priors",
                    "media_costs_by_row_train_scaled",
                    "media_costs_by_row_test_scaled",
                    "extra_features_train_scaled",
                    "extra_features_test_scaled",
                    "target_train_scaled",
                    "target_test_scaled",
                ]
            },
            "display": {
                "date_strs": self.date_strs,
                "media_names": self.media_names,
                "extra_features_names": self.extra_features_names,
            },
            "scalers": {
                "media_scaler": self.media_scaler.to_dict(),
                "media_costs_scaler": self.media_costs_scaler.to_dict(),
                "extra_features_scaler": self.extra_features_scaler.to_dict(),
                "target_scaler": self.target_scaler.to_dict(),
            },
            "config": {
                "has_test_dataset": self.has_test_dataset,
                "target_name": self.target_name,
                "target_is_log_scale": self.target_is_log_scale,
                "time_granularity": self.time_granularity,
            },
        }

    def dump(self, results_dir):
        with gzip.open(os.path.join(results_dir, f"data_to_fit.gz"), "wb") as f:
            msgpack.dump(self.to_dict(), f)

    def to_data_frame(self, unscaled=False):
        """
        :param unscaled: True to unscale the data, False otherwise
        :return: (per-observation df, per-channel df) DataFrames to view DataToFit data.
                 All arrays are deep copies.
        """
        observation_data_by_column_name = {}

        ## media impressions

        if self.has_test_dataset:
            media_data = np.vstack((self.media_data_train_scaled, self.media_data_test_scaled))
        else:
            media_data = self.media_data_train_scaled

        if unscaled:
            media_data = self.media_scaler.inverse_transform(media_data)

        n_media_channels = media_data.shape[1]
        for media_idx in range(n_media_channels):
            col_name = f"{self.media_names[media_idx]} {constants.DATA_FRAME_IMPRESSIONS_SUFFIX}"
            observation_data_by_column_name[col_name] = media_data[:, media_idx]

        ## media costs

        if self.has_test_dataset:
            media_costs_by_row = np.vstack(
                (self.media_costs_by_row_train_scaled, self.media_costs_by_row_test_scaled)
            )
        else:
            media_costs_by_row = self.media_costs_by_row_train_scaled

        if unscaled:
            media_costs_by_row = self.media_costs_scaler.inverse_transform(media_costs_by_row)

        for media_idx in range(n_media_channels):
            col_name = f"{self.media_names[media_idx]} {constants.DATA_FRAME_COST_SUFFIX}"
            observation_data_by_column_name[col_name] = media_costs_by_row[:, media_idx]

        ## extra features

        if self.has_test_dataset:
            extra_features_data = np.vstack(
                (self.extra_features_train_scaled, self.extra_features_test_scaled)
            )
        else:
            extra_features_data = self.extra_features_train_scaled

        if unscaled:
            extra_features_data = self.extra_features_scaler.inverse_transform(extra_features_data)

        for extra_features_idx in range(extra_features_data.shape[1]):
            col_name = self.extra_features_names[extra_features_idx]
            observation_data_by_column_name[col_name] = extra_features_data[:, extra_features_idx]

        ## target data

        if self.has_test_dataset:
            target_data_scaled = np.hstack((self.target_train_scaled, self.target_test_scaled))
        else:
            target_data_scaled = self.target_train_scaled

        if unscaled:
            observation_data_by_column_name[
                self.target_name
            ] = self.target_scaler.inverse_transform(target_data_scaled)
        else:
            observation_data_by_column_name[self.target_name] = target_data_scaled

        # TODO push conversion to datetime upstream so that it is common across all data sets
        per_observation_df = pd.DataFrame(
            data=observation_data_by_column_name,
            index=pd.DatetimeIndex(
                pd.to_datetime(self.date_strs, dayfirst=False, yearfirst=False),
                freq="W" if self.time_granularity == constants.GRANULARITY_WEEKLY else "D",
            ),
            dtype=np.float64,
            copy=True,
        )

        channel_data_by_column_name = {}

        if unscaled:
            channel_data_by_column_name["Cost"] = self.media_costs_scaler.inverse_transform(
                self.media_costs_scaled
            )
            channel_data_by_column_name["Cost Prior"] = self.media_costs_scaler.inverse_transform(
                self.media_cost_priors_scaled
            )
        else:
            channel_data_by_column_name["Cost"] = self.media_costs_scaled
            channel_data_by_column_name["Cost Prior"] = self.media_cost_priors_scaled

        # Learned priors cannot be unscaled
        channel_data_by_column_name["Learned Prior"] = self.learned_media_priors

        per_channel_df = pd.DataFrame(
            data=channel_data_by_column_name, index=self.media_names, dtype=np.float64, copy=True
        )

        return per_observation_df, per_channel_df
