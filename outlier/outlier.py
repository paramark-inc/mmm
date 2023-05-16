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

        outlier_gt_filt = df['values'] > (mean + 2 * stddev)
        if mean > (2 * stddev):
            outlier_lt_filt = df['values'] < (mean - 2 * stddev)
            outliers = df['values'][outlier_gt_filt | outlier_lt_filt]
        else:
            outliers = df['values'][outlier_gt_filt]

        print(f"outlier data points for {metric_name} (mean={mean} stddev={stddev})")
        for i, v in outliers.items():
            print(f"  index={i} value={v}")
