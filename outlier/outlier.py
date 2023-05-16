import pandas as pd


def print_outliers(metric_data):
    """
    print outliers (defined here as any data points more than 2 standard deviations from the mean)
    :param metric_data: dict of form { metric_name : [ metric_values ], }
    """

    for metric_name, metric_values in metric_data.items():
        df = pd.DataFrame({'values': metric_values})
        mean = df['values'].mean()
        stddev = df['values'].std()

        # mean and stddev are floats so this is a floating point operation
        outliers = df['values'][(df['values'] > (mean + 2 * stddev)) | (df['values'] < (mean - 2 * stddev))]

        print(f"outlier data points for {metric_name} (mean={mean:.2f} stddev={stddev:.2f})")
        for i, v in outliers.items():
            print(f"  index={i:4d} value={v:10d}")
