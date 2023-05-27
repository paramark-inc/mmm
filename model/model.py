import numpy as np

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

    def __init__(self,
                 date_strs,
                 time_granularity,
                 media_data,
                 media_costs,
                 media_names,
                 extra_features_data,
                 extra_features_names,
                 target_data,
                 target_name):
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

    def dump(self, verbose=False):
        """
        Debugging routine
        :return:
        """
        print("Dumping input_data")
        if verbose:
            print("\ndate_strs")
            for dstr in self.date_strs:
                print(f"{dstr}")

        print(f"\ntime_granularity={self.time_granularity}")

        print("\nmedia_names")
        for media_name in self.media_names:
            print(f"{media_name}")

        if verbose:
            print("\nmedia_data")
            for media_observation in self.media_data:
                media_line = ""
                for media_val in media_observation:
                    media_line += f"{media_val:,.2f} "
                print(media_line)

        print("\nmedia_costs")
        for media_cost in self.media_costs:
            print(f"{media_cost:,.2f}")

        print("\nextra_features_names")
        for extra_feature_name in self.extra_features_names:
            print(f"{extra_feature_name}")

        if verbose:
            print("\nextra_features_data")
            for extra_feature_observation in self.extra_features_data:
                extra_line = ""
                for extra_feature_val in extra_feature_observation:
                    extra_line += f"{extra_feature_val:,.2f} "
                print(f"{extra_line}")

        print(f"\ntarget_name={self.target_name}")

        if verbose:
            print("\ntarget_data")
            for target in self.target_data:
                print(f"{target:,.2f}")


class DataToFit:
    """
    InputData tranformed to be suitable for fitting a model.  The data undergoes the following transformations
    * split into train and test data set
    * scaled to be smaller values for better accuracy from the Bayesian model
    """

    def __init__(self, media_data_train_scaled, media_data_test_scaled, media_scaler, extra_features_train_scaled,
                 extra_features_test_scaled, extra_features_scaler, media_costs_scaled, media_costs_scaler,
                 target_train_scaled, target_test_scaled, target_scaler):
        self.media_data_train_scaled = media_data_train_scaled
        self.media_data_test_scaled = media_data_test_scaled
        self.media_scaler = media_scaler
        self.extra_features_train_scaled = extra_features_train_scaled
        self.extra_features_test_scaled = extra_features_test_scaled
        self.extra_features_scaler = extra_features_scaler
        self.media_costs_scaled = media_costs_scaled
        self.media_costs_scaler = media_costs_scaler
        self.target_train_scaled = target_train_scaled
        self.target_test_scaled = target_test_scaled
        self.target_scaler = target_scaler
