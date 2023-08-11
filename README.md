## Marketing Mix Models

This repo provides a wrapper around the existing ecosystem of open-source MMM products. It makes it simple to:

* Run MMM from a standard CSV file
* Start with sensible defaults, then experiment further
* Automatically produce a range of useful plots, collated into timestamped directories
<!--- * Remove outliers according to simple rules -->


## Setting up

The easiest way to install dependencies is with a virtual environment:

```
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```

Note that in each new shell you open after this, you'll need to run `source .venv/bin/activate` again to initialize the virtual environment in that shell.


## Quick start

After installing dependencies, use this command to run MMM with the sample data in this repo (generated with Lightweight MMM's [simulator](https://github.com/google/lightweight_mmm/blob/main/lightweight_mmm/utils.py#L80-L165)).

```
python3 run.py -f examples/sample_data.csv -c examples/sample_config.yaml
```


## Preparing your data

Before running MMM -- indeed, before any data science project -- it's important to [tidy](https://cran.r-project.org/web/packages/tidyr/vignettes/tidy-data.html) your dataset.

* All of the data for your model should be in a single CSV file, with one row per time period (day/week).
* There should be a single date index column in ISO format (`YYYY-MM-DD`).
* For each advertising channel, there should be one column for impressions/exposure, and one column for spend.
* There should be a column for your _target metric_, i.e. the KPI that you hope to improve from marketing.
* Optionally, there can be _extra feature_ columns for factors that inflence the KPI separate from marketing.

## Workflow

Once your dataset is ready, your MMM workflow will consist of creating/editing config files, running the code, and then reviewing results.

* **Write a config file** specifying your data format and hyperparameters for the model.
  * In the `data_rows` section, define a `start_date` and `end_date`. This can simply be the first and last dates in your dataset, but you may find that you need to exclude parts of the time range (e.g. if your early data is incomplete).
  * For each of your advertising channels, add an entry in the `media` section, with a display name (used in charts) and the column names for impressions and for spend.
  * Every column in the dataset should be mentioned in the config file; if you don't want to use a column in your model, add it to `ignore_cols`.
  * The other configuration options change Lightweight MMM parameters -- to get started, just copy the defaults from `sample-config.yaml`.
* **Run the code.**
  * `python3 run.py` is the main entrypoint.
  * Use `-f` to pass the path to your CSV data file.
  * Use `-c` to pass the path to your config file.
* **Review results.**
  * By default, output files will be created in a directory called `results/` inside the repo.
  * Start with `model_fit_in_sample.png`, showing model fit for the training set. You don't expect this fit to be perfect, but you do expect the model to pick up trends and be able to predict the general shape of the curve.
  * Model fit for the train/holdout set is shown in `model_fit_out_of_sample.png`.
  * The model's estimate of cost per target unit for each channel is shown in `media_cost_per_target_median.png` and `media_cost_per_target_mean.png`.
  * The attribution chart in `weekly_media_and_baseline_contribution.png` shows how much of the predicted target was attributed by the model to each of your marketing channels, over time.
  * See below for more information about the other files in the `results/` directory.


## Results 

After you run MMM, the `results/` directory will contain the following files:
* `baseline-breakdown.txt`: a breakdown of the components of the baseline curve for each observation in the training data, and an aggregation across all of them.  Exercise caution when interpreting these results, as the uncertainty is not expressed here, and the math is more complicated for baseline components than media channels in some ways.
* `config.yaml`: the config provided when running MMM.
* `data_processed_<channel>_costs.txt`: observation-level spend values per media channel.
* `data_processed_<channel>.txt`: observation-level input values (e.g. impressions) per media channel.
* `data_processed_dates.txt`: mapping of observation index to date.
* `data_processed_<extra_feature>.txt`: observation-level input values per extra feature.
* `data_processed_summary.txt`: summary of input data.
* `data_processed_target.txt`: observation-level target (output) metric data.
* `data_raw_*.txt`: contains data values omitted from the `processed` files above when they are not included in the MMM.  For example, if the raw data is daily, but you are running a weekly MMM, this file will contain the daily data.
* `fit_params.yaml`: parameters passed to the model fitting operation.
* `git_sha.txt`: git hash identifying the code used in the MMM run.
* `media_contribution_mean.png`: shows the mean value of media contribution for each channel.  Media contribution is the fraction of total target metric attributed to a given media channel (e.g. 0.01 for a channel credit with 1% of the total target metric).  Exercise caution when interpreting the mean, because these data sets may be prone to outliers.
* `media_contribution_median.png`: median media contribution per channel.
* `media_cost_per_target_mean.png`: mean cost per target unit, per channel.  Cost per target is the amount of spend required to generate one incremental unit of the target metric.
* `media_cost_per_target_median.png`: median cost per target unit, per channel. 
* `media_performance_breakdown.txt`: blended media effect, roi, and cost per target values as well as channel-level values.  90% credibility intervals are shown in addition to the mean / median.
* `media_roi_mean.png`: mean ROI per channel.  ROI is the number of incremental target metric units for each unit of media spend.
* `media_roi_median.png`: median ROI per channel.
* `metrics_processed.png`: visualization of all input metrics.
* `metrics_raw.png`: visualization of all raw metrics.
* `model_coefficients`: output from the model fit operation, showing the coefficients and associated uncertainty for each parameter.
* `model_fit_in_sample.png`: MAPE and R^2 for the model evaluated against the training data.
* `model_fit_out_of_sample.png`: MAPE and R^2 for the model evaluated against the test data (which is not provided to fit the model).
* `model_media_posteriors.png`: Bayesian posterior distributions of each media parameter.
* `model_priors_and_posteriors.png`: Bayesian priors and posteriors for all model parameters.
* `model.lwm`: Model file that can be provided to `load_model` to perform additional analysis.
* `outliers_processed.txt`: Outlier data points identified and removed.
* `response_curves_cost_per_target.png`:  For each media channel, predicted cost per target at various spend levels, on a per-observation basis.
* `response_curves_target.png`: For each media channel, predicted target metric units generated for spend levels, on a per-observation basis.
* `weekly_media_and_baseline.png`: For each observation in the training period, predicted target metric volume attributed to baseline and media channels.


## Running tests

Simply run `pytest`, excluding tests in the submodules:

```
pytest test/auto/
```


## Further reading

For additional background on running MMM, these articles may be useful:
* Robyn's [analyst guide to MMM](https://facebookexperimental.github.io/Robyn/docs/analysts-guide-to-MMM/)
* Mario Filho's post [How To Create A Marketing Mix Model With LightweightMMM](https://forecastegy.com/posts/how-to-create-a-marketing-mix-model-with-lightweightmmm/)
* Lightweight MMM's [demo notebook](https://github.com/fastrak-inc/lightweight_mmm/blob/main/examples/simple_end_to_end_demo.ipynb)



## Dependencies

Python dependencies are listed in the `requirements.txt` file.

This repository also includes forked copies of these other open-source projects as submodules:
* [lightweight_mmm](https://github.com/google/lightweight_mmm) (Apache License, Version 2.0)
<!---
* [orbit](https://github.com/uber/orbit) (Apache License, Version 2.0)
* [Robyn](https://github.com/facebookexperimental/Robyn) (MIT License)
-->


## License

Copyright 2023 Fastrak, Inc.

Licensed under the Apache License, Version 2.0 (the “License”);
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an “AS IS” BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
