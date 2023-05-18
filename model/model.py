class InputData:
    """
    encapsulation of data fed into the marketing mix model - both the marketing metrics and the sales metrics

    all 2-dimensional arrays have time (day or week number) as the first index and channel as the second index
    all numbers are numpy.uint64, all arrays of numbers are numpy array and numpy ndarray

    all values are true values (i.e. not scaled down for feeding into the MMM)
    """

    def __init__(self, date_strs, time_granularity, media_data, media_costs_per_unit, media_names, extra_features_data,
                 extra_features_names,
                 target_data,
                 target_name):
        """
        :param date_strs: array of labels for each time series data point
        :param time_granularity: integer constant describing the granularity of the time series data
        :param media_data: 2-d array of media data values [time,channel]
        :param media_costs_per_unit: 1-d array of average media costs per unit [channel]
        :param media_names: 1-d array of media channel names
        :param extra_features_data: 2-d array of extra feature values [time, channel]
        :param extra_features_names: 1-d array of extra feature names
        :param target_data: 1-d array of target metric values
        :param target_name: name of target metric
        """
        self.date_strs = date_strs
        self.time_granularity = time_granularity
        self.media_data = media_data
        self.media_costs_per_unit = media_costs_per_unit
        self.media_channel_names = media_names
        self.extra_features_data = extra_features_data
        self.extra_features_names = extra_features_names
        self.target_data = target_data
        self.target_name = target_name
