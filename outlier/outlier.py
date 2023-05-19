import numpy as np
import pandas as pd


def _print_outliers_for_data(name, data):
    df = pd.DataFrame({'values': data})
    mean = df['values'].mean()
    stddev = df['values'].std()

    # mean and stddev are floats so this is a floating point operation
    outliers = df['values'][(df['values'] > (mean + 2 * stddev)) | (df['values'] < (mean - 2 * stddev))]

    print(f"outlier data points for {name} (mean={mean:,.2f} stddev={stddev:,.2f})")
    for i, v in outliers.items():
        stddevs_from_mean = np.absolute(v - mean) / stddev
        print(f"  index={i:4d} value={v:14,.0f} stddevs from mean={stddevs_from_mean:.2f}")


def print_outliers(input_data):
    """
    print outliers (defined here as any data points more than 2 standard deviations from the mean)
    :param input_data: InputData object
    """

    metric_names_and_values = []

    for media_idx in range(input_data.media_data.shape[1]):
        name = f"{input_data.media_names[media_idx]} (volume)"
        metric_names_and_values.append((name, input_data.media_data[:, media_idx]))

    for extra_feature_idx in range(input_data.extra_features_data.shape[1]):
        metric_names_and_values.append(
            (input_data.extra_features_names[extra_feature_idx], input_data.extra_features_data[extra_feature_idx]))

    metric_names_and_values.append((input_data.target_name, input_data.target_data))

    for name, data in metric_names_and_values:
        _print_outliers_for_data(name=name, data=data)
