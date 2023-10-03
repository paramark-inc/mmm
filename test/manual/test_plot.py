import click
import numpy as np
import os
import sys
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "impl", "lightweight_mmm"))

from mmm.constants import constants
from mmm.data import InputData
from mmm.plot.plot import plot_all_metrics


def generate_rand_metric_values(num_observations):
    a = []
    for i in range(num_observations):
        a.append(random.randint(1, 1000))

    return np.array(a, dtype=np.float64)


def test_plot_all_metrics(output_dir):
    num_observations = 10

    metric1_values = generate_rand_metric_values(num_observations)
    metric2_values = generate_rand_metric_values(num_observations)
    metric3_values = generate_rand_metric_values(num_observations)
    metric1_costs = metric1_values * 100.0
    metric2_costs = metric2_values * 200.0
    metric3_costs = metric3_values * 300.0
    target_values = generate_rand_metric_values(num_observations)

    input_data = InputData(
        date_strs=np.full(num_observations, "1/1"),
        time_granularity=constants.GRANULARITY_DAILY,
        media_data=np.column_stack((metric1_values, metric2_values, metric3_values)),
        media_costs=np.array(
            [metric1_costs.sum(), metric2_costs.sum(), metric3_costs.sum()], dtype=np.float64
        ),
        media_costs_by_row=np.column_stack((metric1_costs, metric2_costs, metric3_costs)),
        media_cost_priors=np.array([0.5, 0.4, 0.3], dtype=np.float64),
        media_names=np.array(["metric1", "metric2", "metric3"]),
        extra_features_data=np.ndarray(shape=(0, 0)),
        extra_features_names=np.array([]),
        target_data=target_values,
        target_is_log_scale=False,
        target_name="Sales",
    )
    # input_data.dump()

    plot_all_metrics(input_data, output_dir, "test")


@click.command()
@click.option("--routine", required=True, help="routine name")
@click.option("--output_dir", help="output_dir arg for test_plot_all_metrics")
def run(routine, **kwargs):
    """
    :param routine: routine name
    :param kwargs: routine arguments (provided as kwargs for extensibility)
    :return:
    """
    assert "test_plot_all_metrics" == routine
    assert "output_dir" in kwargs
    test_plot_all_metrics(kwargs["output_dir"])


if __name__ == "__main__":
    run()
