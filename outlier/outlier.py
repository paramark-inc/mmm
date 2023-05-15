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

        outliers = df['values'][df['values'].abs() > (mean + 2 * stddev)]
        print(f"outlier data points for {metric_name} (mean={mean} stddev={stddev})")
        for i, v in outliers.items():
            print(f"  index={i} value={v}")
