import numpy as np
import os
import pandas as pd

from ..constants import constants
from ..impl.lightweight_mmm.lightweight_mmm import preprocessing


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

        assert time_granularity in (constants.GRANULARITY_DAILY, constants.GRANULARITY_WEEKLY), f"{time_granularity}"

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
            target_is_log_scale=input_data.target_is_log_scale,
            target_name=input_data.target_name
        )

    # TODO change dates from strings to numpy datetimes to ensure common formatting
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
            target_is_log_scale,
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
        :param target_is_log_scale: True if target metric is log scale, False otherwise
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
        self.target_is_log_scale = target_is_log_scale

    def dump(self, output_dir, suffix, verbose=False):
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
            summary_file.write(f"\ntarget_is_log_scale={self.target_is_log_scale}\n")

        if verbose:
            with open(os.path.join(output_dir, f"input_data_{suffix}_dates.txt"), "w") as dates_file:
                for idx, dstr in enumerate(self.date_strs):
                    dates_file.write(f"date_strs[{idx:>3}]={dstr:>10}\n")

            for media_idx, media_name in enumerate(self.media_names):
                media_fname = f"input_data_{suffix}_{media_name.lower().replace(' ', '_')}.txt"
                with open(os.path.join(output_dir, media_fname), "w") as media_data_file:
                    for idx, val in enumerate(self.media_data[:, media_idx]):
                        dstr = self.date_strs[idx]
                        media_data_file.write(f"media_data[{idx:>3}][{media_idx}]({dstr:>10})={val:,.2f}\n")

            for extra_features_idx, extra_features_name in enumerate(self.extra_features_names):
                extra_features_fname = f"input_data_{suffix}_{extra_features_name.lower().replace(' ', '_')}.txt"
                with open(os.path.join(output_dir, extra_features_fname), "w") as extra_features_file:
                    for idx, val in enumerate(self.extra_features_data[:, extra_features_idx]):
                        dstr = self.date_strs[idx]
                        extra_features_file.write(
                            f"extra_features_data[{extra_features_idx:>3}][{idx}]({dstr:>10})={val:,.2f}\n"
                        )

            with open(os.path.join(output_dir, f"input_data_{suffix}_target.txt"), "w") as target_file:
                for idx, val in enumerate(self.target_data):
                    dstr = self.date_strs[idx]
                    target_file.write(f"target_data[{idx:>3}]({dstr:>10})={val:,.2f}\n")

    def clone_and_add_extra_features(self, feature_names, feature_data):
        """
        Make a copy of this InputData instance and add an extra feature
        :param feature_names: names to add (list of strings)
        :param feature_data: data to add (2d numpy array of observation, features)
        :return: new InputData instance
        """
        extra_features_names = self.extra_features_names + feature_names
        extra_features_data = np.hstack((self.extra_features_data, feature_data))

        return InputData(
            date_strs=self.date_strs,
            time_granularity=self.time_granularity,
            media_data=self.media_data.copy(),
            media_costs=self.media_costs.copy(),
            media_names=self.media_names.copy(),
            extra_features_data=extra_features_data,
            extra_features_names=extra_features_names,
            target_data=self.target_data.copy(),
            target_is_log_scale=self.target_is_log_scale,
            target_name=self.target_name
        )

    @staticmethod
    def _group_by_week(idx):
        """
        pandas groupby callback to use for grouping rows into groups of 7
        :param idx: row index
        :return: group index
        """
        return idx // 7

    def clone_as_weekly(self):
        """
        Compress from daily form to weekly form.  The data will be grouped into groups of 7 rows, starting with the
        first 7 rows, and discarding the last group of less than 7 rows.  This means that the groups are aligned to
        the day of week for the first row, not Sunday.

        :return: new InputData instance, with weekly data
        """
        assert self.time_granularity == constants.GRANULARITY_DAILY

        # we need to trim one partial week off the end if the data is not an even number of weeks
        needs_cut_last = False if self.media_data.shape[0] % 7 == 0 else True

        date_strs_weekly = self.date_strs.copy()[::7]

        media_data_dict = {name: self.media_data[:, idx] for idx, name in enumerate(self.media_names)}
        media_df_daily = pd.DataFrame(data=media_data_dict)
        media_df_weekly = media_df_daily.groupby(by=InputData._group_by_week).sum()

        extra_features_data_dict = {name: self.extra_features_data[:, idx] for idx, name in
                                    enumerate(self.extra_features_names)}
        extra_features_df_daily = pd.DataFrame(data=extra_features_data_dict)
        extra_features_df_weekly = extra_features_df_daily.groupby(by=InputData._group_by_week).sum()

        target_data_dict = {self.target_name: self.target_data}
        target_df_daily = pd.DataFrame(data=target_data_dict)
        target_df_weekly = target_df_daily.groupby(by=InputData._group_by_week).sum()

        if needs_cut_last:
            date_strs_weekly = date_strs_weekly[:-1]
            media_df_weekly = media_df_weekly[:-1]
            extra_features_df_weekly = extra_features_df_weekly[:-1]
            target_df_weekly = target_df_weekly[:-1]

        extra_features_data = (
            extra_features_df_weekly.to_numpy()
            if extra_features_df_weekly.shape[1]
            else np.ndarray(shape=(media_df_weekly.shape[0], 0), dtype=np.float64)
        )

        return InputData(
            date_strs=date_strs_weekly,
            time_granularity=constants.GRANULARITY_WEEKLY,
            media_data=media_df_weekly.to_numpy(),
            media_costs=self.media_costs.copy(),
            media_names=self.media_names.copy(),
            extra_features_data=extra_features_data,
            extra_features_names=self.extra_features_names.copy(),
            # DataFrame returns a 2D array even when there's only one column
            target_data=target_df_weekly.to_numpy()[:, 0],
            target_is_log_scale=self.target_is_log_scale,
            target_name=self.target_name,
        )

    def clone_and_log_transform_target_data(self):
        """
        clone this input_data and log transform the target data.  Note that charts will show the log-transformed values.
        You'll need to exp() them manually.
        :return: new InputData instance
        """
        assert not self.target_is_log_scale

        return InputData(
            date_strs=self.date_strs.copy(),
            time_granularity=self.time_granularity,
            media_data=self.media_data.copy(),
            media_costs=self.media_costs.copy(),
            media_names=self.media_names.copy(),
            extra_features_data=self.extra_features_data.copy(),
            extra_features_names=self.extra_features_names.copy(),
            target_data=np.log(self.target_data),
            target_is_log_scale=True,
            target_name=self.target_name + " (log-transformed)"
        )


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

        split_point = data_size - data_size // 10
        media_data_train = input_data.media_data[:split_point, :]
        media_data_test = input_data.media_data[split_point:, :]
        target_train = input_data.target_data[:split_point]
        target_test = input_data.target_data[split_point:]
        extra_features_train = input_data.extra_features_data[:split_point, :]
        extra_features_test = input_data.extra_features_data[split_point:, :]

        # Scale data (ignoring the zeroes in the media data).  Call fit_transform only the first time because only one
        # scaling constant is stored in the scaler.
        media_scaler = preprocessing.CustomScaler(divide_operation=DataToFit._robust_scaling_divide_operation)
        extra_features_scaler = preprocessing.CustomScaler(divide_operation=DataToFit._robust_scaling_divide_operation)
        target_scaler = preprocessing.CustomScaler(divide_operation=DataToFit._robust_scaling_divide_operation)

        # scale cost up by N since fit() will divide it by number of time periods
        media_cost_scaler = preprocessing.CustomScaler(divide_operation=DataToFit._robust_scaling_divide_operation)

        media_data_train_scaled = media_scaler.fit_transform(media_data_train)
        media_data_test_scaled = media_scaler.transform(media_data_test)
        extra_features_train_scaled = extra_features_scaler.fit_transform(extra_features_train)
        extra_features_test_scaled = extra_features_scaler.transform(extra_features_test)
        target_train_scaled = target_scaler.fit_transform(target_train)
        target_test_scaled = target_scaler.transform(target_test)

        # lightweightMMM requires that media priors are > 0 by virtue of using HalfNormal which has a Positive
        # constraint on all values.
        for idx, cost in np.ndenumerate(input_data.media_costs):
            assert cost > 0.0, f"Media channel {idx[0]} has zero cost"
        media_costs_scaled = media_cost_scaler.fit_transform(input_data.media_costs)

        return DataToFit(
            date_strs=input_data.date_strs,
            time_granularity=input_data.time_granularity,
            media_data_train_scaled=media_data_train_scaled,
            media_data_test_scaled=media_data_test_scaled,
            media_scaler=media_scaler,
            media_costs_scaled=media_costs_scaled,
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
            target_name=input_data.target_name
        )

    def __init__(
            self,
            date_strs,
            time_granularity,
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
            target_is_log_scale,
            target_scaler,
            target_name
    ):
        self.date_strs = date_strs
        self.time_granularity = time_granularity
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
        for media_idx in range(media_data_scaled.shape[1]):
            col_name = f"{self.media_names[media_idx]} volume"
            media_data_touse = media_data_unscaled if unscaled else media_data_scaled
            observation_data_by_column_name[col_name] = media_data_touse[:, media_idx]

        extra_features_data_scaled = np.vstack((self.extra_features_train_scaled, self.extra_features_test_scaled))
        extra_features_data_unscaled = self.extra_features_scaler.inverse_transform(extra_features_data_scaled)
        for extra_features_idx in range(extra_features_data_scaled.shape[1]):
            col_name = self.extra_features_names[extra_features_idx]
            extra_features_data_touse = extra_features_data_unscaled if unscaled else extra_features_data_scaled
            observation_data_by_column_name[col_name] = extra_features_data_touse[:, extra_features_idx]

        target_data_scaled = np.hstack((self.target_train_scaled, self.target_test_scaled))
        target_data_unscaled = self.target_scaler.inverse_transform(target_data_scaled)
        target_data_touse = target_data_unscaled if unscaled else target_data_scaled
        observation_data_by_column_name[self.target_name] = target_data_touse

        # TODO push conversion to datetime upstream so that it is common across all data sets
        per_observation_df = pd.DataFrame(
            data=observation_data_by_column_name,
            index=pd.to_datetime(self.date_strs, dayfirst=False, yearfirst=False),
            dtype=np.float64,
            copy=True
        )

        media_costs_unscaled = self.media_costs_scaler.inverse_transform(self.media_costs_scaled)
        media_costs_touse = media_costs_unscaled if unscaled else self.media_costs_scaled
        channel_data_by_column_name = {"Cost": media_costs_touse}

        per_channel_df = pd.DataFrame(
            data=channel_data_by_column_name,
            index=self.media_names,
            dtype=np.float64,
            copy=True
        )

        return per_observation_df, per_channel_df
