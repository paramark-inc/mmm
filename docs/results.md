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