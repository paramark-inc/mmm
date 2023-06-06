import numpy as np
import os
import pandas as pd

from ..constants import constants


class InputData:
    """
    encapsulation of data fed into the marketing mix model - both the marketing metrics and the sales metrics

    all 2-dimensional arrays have time (day or week number) as the first index and channel as the second index
    all numbers are numpy.uint64, all arrays of numbers are numpy array and numpy ndarray

    all values are true values (i.e. not scaled down for feeding into the MMM)
    """

    @staticmethod
    def _validate(date_strs,
                  time_granularity,
                  media_data,
                  media_costs,
                  media_names,
                  extra_features_data,
                  extra_features_names,
                  target_data,
                  target_name):
        num_observations = date_strs.shape[0]

        assert time_granularity in (constants.GRANULARITY_DAILY, constants.GRANULARITY_DAILY), f"{time_granularity}"

        assert 2 == media_data.ndim, f"{media_data.ndim}"
        assert num_observations == media_data.shape[0], f"{num_observations} {media_data.shape[0]}"
        num_channels = media_data.shape[1]
        assert np.float64 == media_data.dtype, f"{np.float64} {media_data.dtype}"

        assert 1 == media_costs.ndim, f"{media_costs.ndim}"
        assert num_channels == media_costs.shape[0], f"{num_channels} {media_costs.shape[0]}"
        assert np.float64 == media_costs.dtype, f"{np.float64} {media_costs.dtype}"

        assert num_channels == len(media_names), f"{num_channels} {len(media_names)}"

        assert 2 == extra_features_data.ndim, f"{extra_features_data.ndim}"
        num_extra_features = extra_features_data.shape[1]
        if num_extra_features:
            assert num_observations == extra_features_data.shape[
                0], f"{num_observations} {extra_features_data.shape[0]}"

        assert num_extra_features == len(extra_features_names), f"{num_extra_features} {len(extra_features_names)}"

        assert 1 == target_data.ndim, f"{target_data.ndim}"
        assert num_observations == target_data.shape[0], f"{num_observations} {target_data.shape[0]}"
        assert np.float64 == target_data.dtype, f"{np.float64} {target_data.dtype}"

        assert target_name

    @staticmethod
    def clone_with_data_edits(input_data, editor_func, context):
        """
        clone the input_data while allowing the editor_func to modify data values on a copy of the data.

        :param input_data: InputData instance
        :param editor_func: function called to allow editing of copies
          args: context, date_strs, media_data, extra_features_data, target_data
                 (each is a copy that the func can modify)
          return: none
        :param context client context
        :return: new InputData instance (deep copy)
        """
        date_strs_copy = input_data.date_strs.copy()
        media_data_copy = input_data.media_data.copy()
        extra_features_data_copy = input_data.extra_features_data.copy()
        target_data_copy = input_data.target_data.copy()

        editor_func(
            context=context,
            date_strs=date_strs_copy,
            media_data=media_data_copy,
            extra_features_data=extra_features_data_copy,
            target_data=target_data_copy
        )

        return InputData(
            date_strs=date_strs_copy,
            time_granularity=input_data.time_granularity,
            media_data=media_data_copy,
            media_costs=input_data.media_costs.copy(),
            media_names=input_data.media_names.copy(),
            extra_features_data=extra_features_data_copy,
            extra_features_names=input_data.extra_features_names.copy(),
            target_data=target_data_copy,
            target_name=input_data.target_name
        )

    def __init__(
            self,
            date_strs,
            time_granularity,
            media_data,
            media_costs,
            media_names,
            extra_features_data,
            extra_features_names,
            target_data,
            target_name
    ):
        """
        :param date_strs: 1-d numpy array of labels for each time series data point
        :param time_granularity: string constant describing the granularity of the time series data (
                                 constants.GRANULARITY_DAILY, constants.GRANULARITY_WEEKLY, etc.)
        :param media_data: 2-d numpy array of float64 media data values [time,channel]
        :param media_costs: 1-d numpy array of float64 total media costs [channel]
        :param media_names: list of media channel names
        :param extra_features_data: 2-d numpy array of float64 extra feature values [time, channel]
        :param extra_features_names: list of extra feature names
        :param target_data: 1-d numpy array of float64 target metric values
        :param target_name: name of target metric
        """
        InputData._validate(date_strs=date_strs, time_granularity=time_granularity, media_data=media_data,
                            media_costs=media_costs, media_names=media_names,
                            extra_features_data=extra_features_data, extra_features_names=extra_features_names,
                            target_data=target_data, target_name=target_name)

        self.date_strs = date_strs
        self.time_granularity = time_granularity
        self.media_data = media_data
        self.media_costs = media_costs
        self.media_names = media_names
        self.extra_features_data = extra_features_data
        self.extra_features_names = extra_features_names
        self.target_data = target_data
        self.target_name = target_name

    def dump(self, output_dir, suffix, verbose=True):
        """
        Debugging routine
        :param output_dir: path to output directory to write to
        :param suffix: suffix to append to filename
        :param verbose True to get verbose printing
        :return:
        """

        with open(os.path.join(output_dir, f"input_data_{suffix}_summary.txt"), "w") as summary_file:
            summary_file.write(f"time_granularity={self.time_granularity}\n")
            summary_file.write(f"\nmedia_names:\n")
            for idx, media_name in enumerate(self.media_names):
                summary_file.write(f"media_names[{idx}]={media_name}\n")
            summary_file.write(f"\nextra_features_names:\n")
            for idx, extra_features_name in enumerate(self.extra_features_names):
                summary_file.write(f"extra_features_names[{idx}]={extra_features_name}\n")
            summary_file.write("\nmedia_costs:\n")
            for idx, media_cost in enumerate(self.media_costs):
                summary_file.write(f"media_costs[{idx}]={media_cost:,.2f}\n")
            summary_file.write(f"\ntarget_name={self.target_name}\n")

        if not verbose:
            with open(os.path.join(output_dir, f"input_data_{suffix}_dates.txt"), "w") as dates_file:
                for idx, dstr in enumerate(self.date_strs):
                    dates_file.write(f"date_strs[{idx:>3}]={dstr:>5}\n")

            for media_idx, media_name in enumerate(self.media_names):
                media_fname = f"input_data_{suffix}_{media_name.lower()}.txt"
                with open(os.path.join(output_dir, media_fname), "w") as media_data_file:
                    for idx, val in enumerate(self.media_data[:, media_idx]):
                        media_data_file.write(f"media_data[{idx:>3}][{media_idx}]={val:,.2f}\n")

            for extra_features_idx, extra_features_name in enumerate(self.extra_features_names):
                extra_features_fname = f"input_data_{suffix}_{extra_features_name.lower()}.txt"
                with open(os.path.join(output_dir, extra_features_fname), "w") as extra_features_file:
                    for idx, val in enumerate(self.extra_features_data[:, extra_features_idx]):
                        extra_features_file.write(f"extra_features_data[{extra_features_idx:>3}][{idx}]={val:,.2f}\n")

            with open(os.path.join(output_dir, f"input_data_{suffix}_target.txt"), "w") as target_file:
                for idx, val in enumerate(self.target_data):
                    target_file.write(f"target_data[{idx:>3}]={val:,.2f}\n")


class DataToFit:
    """
    InputData tranformed to be suitable for fitting a model.  The data undergoes the following transformations
    * split into train and test data set
    * scaled to be smaller values for better accuracy from the Bayesian model
    """

    def __init__(
            self,
            date_strs,
            media_data_train_scaled,
            media_data_test_scaled,
            media_scaler,
            media_costs_scaled,
            media_costs_scaler,
            media_names,
            extra_features_train_scaled,
            extra_features_test_scaled,
            extra_features_scaler,
            extra_features_names,
            target_train_scaled,
            target_test_scaled,
            target_scaler,
            target_name
    ):
        self.date_strs = date_strs
        self.media_data_train_scaled = media_data_train_scaled
        self.media_data_test_scaled = media_data_test_scaled
        self.media_scaler = media_scaler
        self.media_costs_scaled = media_costs_scaled
        self.media_costs_scaler = media_costs_scaler
        self.media_names = media_names
        self.extra_features_train_scaled = extra_features_train_scaled
        self.extra_features_test_scaled = extra_features_test_scaled
        self.extra_features_scaler = extra_features_scaler
        self.extra_features_names = extra_features_names
        self.target_train_scaled = target_train_scaled
        self.target_test_scaled = target_test_scaled
        self.target_scaler = target_scaler
        self.target_name = target_name

    def to_data_frame(self):
        """
        :return: (per-observation df, per-channel df) DataFrames to view DataToFit data.
                 All arrays are deep copies.
        """
        observation_data_by_column_name = {}

        media_data = np.vstack((self.media_data_train_scaled, self.media_data_test_scaled))
        for media_idx in range(media_data.shape[1]):
            col_name = f"{self.media_names[media_idx]} volume"
            observation_data_by_column_name[col_name] = media_data[:, media_idx]

        extra_features_data = np.vstack((self.extra_features_train_scaled, self.extra_features_test_scaled))
        for extra_features_idx in range(extra_features_data.shape[1]):
            col_name = self.extra_features_names[extra_features_idx]
            observation_data_by_column_name[col_name] = extra_features_data[:, extra_features_idx]

        target_data = np.hstack((self.target_train_scaled, self.target_test_scaled))
        observation_data_by_column_name[self.target_name] = target_data

        # TODO push conversion to datetime upstream so that it is common across all data sets
        per_observation_df = pd.DataFrame(
            data=observation_data_by_column_name,
            index=pd.to_datetime(self.date_strs, dayfirst=False, yearfirst=False),
            dtype=np.float64,
            copy=True
        )

        channel_data_by_column_name = {"Cost": self.media_costs_scaled}

        per_channel_df = pd.DataFrame(
            data=channel_data_by_column_name,
            index=self.media_names,
            dtype=np.float64,
            copy=True
        )

        return per_observation_df, per_channel_df
