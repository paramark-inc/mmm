import numpy as np
import os
import pandas as pd

from mmm.constants import constants


class InputData:
    """
    encapsulation of data fed into the marketing mix model - both the marketing metrics and the sales metrics

    all 2-dimensional arrays have time (day or week number) as the first index and channel as the second index
    all numbers are numpy.uint64, all arrays of numbers are numpy array and numpy ndarray

    all values are true values (i.e. not scaled down for feeding into the MMM)
    """

    @staticmethod
    def _validate(
        date_strs,
        time_granularity,
        media_data,
        media_costs,
        media_cost_priors,
        learned_media_priors,
        media_names,
        extra_features_data,
        extra_features_names,
        target_data,
        target_name,
    ):
        num_observations = date_strs.shape[0]

        assert time_granularity in (
            constants.GRANULARITY_DAILY,
            constants.GRANULARITY_WEEKLY,
        ), f"{time_granularity}"

        assert 2 == media_data.ndim, f"{media_data.ndim}"
        assert num_observations == media_data.shape[0], f"{num_observations} {media_data.shape[0]}"
        num_channels = media_data.shape[1]
        assert np.float64 == media_data.dtype, f"{np.float64} {media_data.dtype}"

        assert 1 == media_costs.ndim, f"{media_costs.ndim}"
        assert num_channels == media_costs.shape[0], f"{num_channels} {media_costs.shape[0]}"
        assert np.float64 == media_costs.dtype, f"{np.float64} {media_costs.dtype}"

        assert 1 == media_cost_priors.ndim, f"{media_cost_priors.ndim}"
        assert (
            num_channels == media_cost_priors.shape[0]
        ), f"{num_channels} {media_cost_priors.shape[0]}"
        assert np.float64 == media_cost_priors.dtype, f"{np.float64} {media_cost_priors.dtype}"
        # lightweightMMM requires that media priors are > 0 by virtue of using HalfNormal which has a Positive
        # constraint on all values.
        for idx, prior in np.ndenumerate(media_cost_priors):
            assert (
                prior > 0.0 or learned_media_priors[idx] > 0.0
            ), f"Media channel {media_names[idx[0]]} has a zero cost prior and no learned prior was specified. Make sure this channel's cost column has non-zero and non-NaN values."

        assert num_channels == len(media_names), f"{num_channels} {len(media_names)}"

        assert 2 == extra_features_data.ndim, f"{extra_features_data.ndim}"
        num_extra_features = extra_features_data.shape[1]
        if num_extra_features:
            assert (
                num_observations == extra_features_data.shape[0]
            ), f"{num_observations} {extra_features_data.shape[0]}"

        assert num_extra_features == len(
            extra_features_names
        ), f"{num_extra_features} {len(extra_features_names)}"

        assert 1 == target_data.ndim, f"{target_data.ndim}"
        assert (
            num_observations == target_data.shape[0]
        ), f"{num_observations} {target_data.shape[0]}"
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
            target_data=target_data_copy,
        )

        return InputData(
            date_strs=date_strs_copy,
            time_granularity=input_data.time_granularity,
            media_data=media_data_copy,
            media_costs=input_data.media_costs.copy(),
            media_costs_by_row=input_data.media_costs_by_row.copy(),
            media_cost_priors=input_data.media_cost_priors.copy(),
            learned_media_priors=input_data.learned_media_priors.copy(),
            media_names=input_data.media_names.copy(),
            extra_features_data=extra_features_data_copy,
            extra_features_names=input_data.extra_features_names.copy(),
            target_data=target_data_copy,
            target_is_log_scale=input_data.target_is_log_scale,
            target_name=input_data.target_name,
        )

    # TODO change dates from strings to numpy datetimes to ensure common formatting
    def __init__(
        self,
        date_strs,
        time_granularity,
        media_data,
        media_costs,
        media_costs_by_row,
        media_cost_priors,
        learned_media_priors,
        media_names,
        extra_features_data,
        extra_features_names,
        target_data,
        target_is_log_scale,
        target_name,
    ):
        """
        :param date_strs: 1-d numpy array of labels for each time series data point
        :param time_granularity: string constant describing the granularity of the time series data (
                                 constants.GRANULARITY_DAILY, constants.GRANULARITY_WEEKLY, etc.)
        :param media_data: 2-d numpy array of float64 media data values [time,channel]
        :param media_costs: 1-d numpy array of float64 total media costs [channel]
        :param media_costs_by_row: 2-d numpy array of float 64 media costs per day [time, channel]
        :param media_cost_priors: 1-d numpy array of float64 media prior [channel].  For most forms of paid media this will
                             be equivalent to the costs.  However, in cases where the actual cost is zero or very small,
                             it makes sense to use a different value as the prior.
        :param learned_media_priors: 1-d array of float64 media prior [channel].  These priors will override the cost
                             priors when provided (i.e. > 0.).  These may be informed by an experiment or
                             an MMM run on an earlier period.  These values will be provided directly to
                             LightweightMMM without any scaling.
        :param media_names: list of media channel names
        :param extra_features_data: 2-d numpy array of float64 extra feature values [time, channel]
        :param extra_features_names: list of extra feature names
        :param target_data: 1-d numpy array of float64 target metric values
        :param target_is_log_scale: True if target metric is log scale, False otherwise
        :param target_name: name of target metric
        """
        InputData._validate(
            date_strs=date_strs,
            time_granularity=time_granularity,
            media_data=media_data,
            media_costs=media_costs,
            media_cost_priors=media_cost_priors,
            learned_media_priors=learned_media_priors,
            media_names=media_names,
            extra_features_data=extra_features_data,
            extra_features_names=extra_features_names,
            target_data=target_data,
            target_name=target_name,
        )

        self.date_strs = date_strs
        self.time_granularity = time_granularity
        self.media_data = media_data
        self.media_costs = media_costs
        self.media_costs_by_row = media_costs_by_row
        self.media_cost_priors = media_cost_priors
        self.learned_media_priors = learned_media_priors
        self.media_names = media_names
        self.extra_features_data = extra_features_data
        self.extra_features_names = extra_features_names
        self.target_data = target_data
        self.target_name = target_name
        self.target_is_log_scale = target_is_log_scale

    @staticmethod
    def _sanitize_name(media_name):
        """
        Sanitize a channel or extra feature name for the purpose of creating filenames.
        :param media_name: channel name
        :return: sanititized name
        """
        return media_name.lower().replace(" ", "_").replace("(", "").replace(")", "")

    def dump(self, output_dir, suffix, verbose=False):
        """
        Debugging routine
        :param output_dir: path to output directory to write to
        :param suffix: suffix to append to filename
        :param verbose True to get verbose printing
        :return:
        """

        with open(os.path.join(output_dir, f"data_{suffix}_summary.txt"), "w") as summary_file:
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
            summary_file.write("\nmedia_cost_priors:\n")
            for idx, media_prior in enumerate(self.media_cost_priors):
                summary_file.write(f"media_priors[{idx}]={media_prior:,.2f}\n")
            summary_file.write("\nmedia_cost_priors:\n")
            for idx, learned_prior in enumerate(self.learned_media_priors):
                summary_file.write(f"learned_media_priors[{idx}]={learned_prior:,.2f}\n")
            summary_file.write(f"\ntarget_name={self.target_name}\n")
            summary_file.write(f"\ntarget_is_log_scale={self.target_is_log_scale}\n")

        if verbose:
            with open(os.path.join(output_dir, f"data_{suffix}_dates.txt"), "w") as dates_file:
                for idx, dstr in enumerate(self.date_strs):
                    dates_file.write(f"date_strs[{idx:>3}]={dstr:>10}\n")

            for media_idx, media_name in enumerate(self.media_names):
                media_fname = f"data_{suffix}_{InputData._sanitize_name(media_name)}.txt"
                with open(os.path.join(output_dir, media_fname), "w") as media_data_file:
                    for idx, val in enumerate(self.media_data[:, media_idx]):
                        dstr = self.date_strs[idx]
                        media_data_file.write(
                            f"media_data[{idx:>3}][{media_idx}]({dstr:>10})={val:,.2f}\n"
                        )

            for media_idx, media_name in enumerate(self.media_names):
                media_fname = f"data_{suffix}_{InputData._sanitize_name(media_name)}_costs.txt"
                with open(os.path.join(output_dir, media_fname), "w") as media_costs_file:
                    for idx, val in enumerate(self.media_costs_by_row[:, media_idx]):
                        dstr = self.date_strs[idx]
                        media_costs_file.write(
                            f"media_costs_by_row[{idx:>3}][{media_idx}]({dstr:>10})={val:,.2f}\n"
                        )

            for extra_features_idx, extra_features_name in enumerate(self.extra_features_names):
                extra_features_fname = (
                    f"data_{suffix}_{InputData._sanitize_name(extra_features_name)}.txt"
                )
                with open(
                    os.path.join(output_dir, extra_features_fname), "w"
                ) as extra_features_file:
                    for idx, val in enumerate(self.extra_features_data[:, extra_features_idx]):
                        dstr = self.date_strs[idx]
                        extra_features_file.write(
                            f"extra_features_data[{extra_features_idx:>3}][{idx}]({dstr:>10})={val:,.2f}\n"
                        )

            with open(os.path.join(output_dir, f"data_{suffix}_target.txt"), "w") as target_file:
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
            media_costs_by_row=self.media_costs_by_row.copy(),
            media_cost_priors=self.media_cost_priors.copy(),
            learned_media_priors=self.learned_media_priors.copy(),
            media_names=self.media_names.copy(),
            extra_features_data=extra_features_data,
            extra_features_names=extra_features_names,
            target_data=self.target_data.copy(),
            target_is_log_scale=self.target_is_log_scale,
            target_name=self.target_name,
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

        media_data_dict = {
            name: self.media_data[:, idx] for idx, name in enumerate(self.media_names)
        }
        media_df_daily = pd.DataFrame(data=media_data_dict)
        media_df_weekly = media_df_daily.groupby(by=InputData._group_by_week).sum()

        media_costs_data_dict = {
            name: self.media_costs_by_row[:, idx] for idx, name in enumerate(self.media_names)
        }
        media_costs_df_daily = pd.DataFrame(data=media_costs_data_dict)
        media_costs_df_weekly = media_costs_df_daily.groupby(by=InputData._group_by_week).sum()

        extra_features_data_dict = {
            name: self.extra_features_data[:, idx]
            for idx, name in enumerate(self.extra_features_names)
        }
        extra_features_df_daily = pd.DataFrame(data=extra_features_data_dict)
        extra_features_df_weekly = extra_features_df_daily.groupby(
            by=InputData._group_by_week
        ).sum()

        target_data_dict = {self.target_name: self.target_data}
        target_df_daily = pd.DataFrame(data=target_data_dict)
        target_df_weekly = target_df_daily.groupby(by=InputData._group_by_week).sum()

        if needs_cut_last:
            date_strs_weekly = date_strs_weekly[:-1]
            media_df_weekly = media_df_weekly[:-1]
            media_costs_df_weekly = media_costs_df_weekly[:-1]
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
            media_costs_by_row=media_costs_df_weekly.to_numpy(),
            media_cost_priors=self.media_cost_priors.copy(),
            learned_media_priors=self.learned_media_priors.copy(),
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
        return InputData(
            date_strs=self.date_strs.copy(),
            time_granularity=self.time_granularity,
            media_data=self.media_data.copy(),
            media_costs=self.media_costs.copy(),
            media_costs_by_row=self.media_costs_by_row.copy(),
            media_cost_priors=self.media_cost_priors.copy(),
            learned_media_priors=self.learned_media_priors.copy(),
            media_names=self.media_names.copy(),
            extra_features_data=self.extra_features_data.copy(),
            extra_features_names=self.extra_features_names.copy(),
            target_data=np.log(self.target_data),
            target_is_log_scale=True,
            target_name=self.target_name + " (log-transformed)",
        )

    def clone_and_split_media_data(
        self, channel_idx, split_obs_idx, media_before_name, media_after_name
    ):
        """
        clone this input_data and split the given media column around a split point.  "Split" means to create
        a new media column, place the data starting at 'media_idx' in the new column, and fill empty values with
        zeroes.

        :param channel_idx: media channel to split into two
        :param split_obs_idx: observation index to begin the new "after" channel at
        :param media_before_name: name of the "before" channel created by this function
        :param media_after_name: name of the "after" channel created by this function
        :return:
        """
        media_names = (
            self.media_names[0:channel_idx]
            + [media_before_name, media_after_name]
            + self.media_names[channel_idx + 1 :]
        )

        media_data_before_column = self.media_data[:, channel_idx].copy()
        media_data_before_column[split_obs_idx:] = 0.0
        media_data_after_column = self.media_data[:, channel_idx].copy()
        media_data_after_column[:split_obs_idx] = 0.0

        media_data = np.zeros(shape=(self.media_data.shape[0], self.media_data.shape[1] + 1))
        media_data[:, :channel_idx] = self.media_data[:, :channel_idx]
        media_data[:, channel_idx] = media_data_before_column
        media_data[:, channel_idx + 1] = media_data_after_column
        media_data[:, channel_idx + 2 :] = self.media_data[:, channel_idx + 1 :]

        media_costs_by_row_before_column = self.media_costs_by_row[:, channel_idx].copy()
        media_costs_by_row_before_column[split_obs_idx:] = 0.0
        media_costs_by_row_after_column = self.media_costs_by_row[:, channel_idx].copy()
        media_costs_by_row_after_column[:split_obs_idx] = 0.0

        media_costs_by_row = np.zeros(
            shape=(self.media_costs_by_row.shape[0], self.media_costs_by_row.shape[1] + 1)
        )
        media_costs_by_row[:, :channel_idx] = self.media_costs_by_row[:, :channel_idx]
        media_costs_by_row[:, channel_idx] = media_costs_by_row_before_column
        media_costs_by_row[:, channel_idx + 1] = media_costs_by_row_after_column
        media_costs_by_row[:, channel_idx + 2 :] = self.media_costs_by_row[:, channel_idx + 1 :]

        media_costs = np.zeros(shape=(self.media_costs.shape[0] + 1,))
        media_costs[:channel_idx] = self.media_costs[:channel_idx]
        media_costs[channel_idx] = media_costs_by_row_before_column.sum()
        media_costs[channel_idx + 1] = media_costs_by_row_after_column.sum()
        media_costs[channel_idx + 2 :] = self.media_costs[channel_idx + 1 :]

        media_cost_priors = np.zeros(shape=(self.media_cost_priors.shape[0] + 1,))
        media_cost_priors[:channel_idx] = self.media_cost_priors[:channel_idx]
        # artificially scale media_cost_priors down by the percentage of observations that are included in the split
        # column.  This is technically incorrect, but since priors do not directly impact the results, presumably
        # good enough.
        split_point_pct = split_obs_idx / self.media_data.shape[0]
        media_cost_priors[channel_idx] = self.media_cost_priors[channel_idx] * split_point_pct
        media_cost_priors[channel_idx + 1] = self.media_cost_priors[channel_idx] * (
            1 - split_point_pct
        )
        media_cost_priors[channel_idx + 2 :] = self.media_cost_priors[channel_idx + 1 :]

        return InputData(
            date_strs=self.date_strs.copy(),
            time_granularity=self.time_granularity,
            media_data=media_data,
            media_costs=media_costs,
            media_costs_by_row=media_costs_by_row,
            media_cost_priors=media_cost_priors,
            learned_media_priors=self.learned_media_priors.copy(),
            media_names=media_names,
            extra_features_data=self.extra_features_data.copy(),
            extra_features_names=self.extra_features_names.copy(),
            target_data=self.target_data.copy(),
            target_is_log_scale=self.target_is_log_scale,
            target_name=self.target_name,
        )
