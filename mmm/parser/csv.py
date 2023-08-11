from mmm.constants import constants

# noinspection PyPackageRequirements
import numpy as np
import pandas as pd


def parse_csv_generic(data_fname: str, config: dict):
    """
    :param data_fname: full path to file for raw data
    :param config: config dictionary (from yaml file)
    :return: dict with format {
            KEY_GRANULARITY: granularity,
            KEY_OBSERVATIONS: observations,
            KEY_METRICS: { metric_name: [ metric values ], ... }
        }
    """
    # allow config option for name of date column;
    # if not provided, use "date" (case sensitive)
    index_col = config.get("date_col", "date")

    data_df = pd.read_csv(data_fname, index_col=index_col)

    if "total" in config.get("data_rows", {}):
        total_rows_expected = config["data_rows"]["total"]
        assert (
            total_rows_expected == data_df.shape[0]
        ), f"lhs={total_rows_expected}, rhs={data_df.shape[0]}"

    if "start_date" in config["data_rows"] and "end_date" in config["data_rows"]:
        start = config["data_rows"]["start_date"].isoformat()
        end = config["data_rows"]["end_date"].isoformat()
        data_df = data_df.loc[start:end]
    elif "to_use" in config["data_rows"]:
        data_df = data_df.tail(config["data_rows"]["to_use"])

    if "ignore_cols" in config:
        data_df = data_df.drop(columns=config["ignore_cols"])

    date_strs = data_df.index.to_numpy()

    metric_dict = {}
    for column in data_df.columns:
        metric_dict[column] = data_df[column].to_numpy(dtype=np.float64)

    data_dict = {
        constants.KEY_GRANULARITY: config.get("raw_data_granularity"),
        constants.KEY_OBSERVATIONS: data_df.shape[0],
        constants.KEY_DATE_STRS: date_strs,
        constants.KEY_METRICS: metric_dict,
    }

    return data_dict
