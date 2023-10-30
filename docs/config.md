## Config

To run MMM, you'll need to create a `yaml` config file with the following parameters:
* `library`: at present only `lightweight_mmm` is supported.
* `model_name`: specifies the underlying mathematical formula used by the model.  One of `adstock`, `hill_adstock`, and `carryover`.
* `number_warmup`: (optional) number of samples taken during the warm up phase.  These samples will be discarded.  We recommend a value of at least 1000. Defaults to 2000.
* `number_samples`: (optional) number of samples taken during the modeling phase.  These samples will be used by the model.  We recommend a value of at least 1000.Defaults to 2000.
* `number_chains`: (optional) number of chains to sample on.  Valid values range from 1 to the number of CPU cores on the system.  Running with multiple chains allows the model to perform a more robust test of the sampling accuracy / repeatability.  This comes at the cost of a longer runtime. Defaults to 1.
* `degrees_seasonality`: (optional) integer number of degrees of seasonality.  Larger numbers indicate more nested seasonality effect.  We recommend starting with 1, 2, or 3. Defaults to 2.
* `weekday_seasonality`: (optional) controls how the model handles day of week changes. Use `null` to automatically derive the value from the time_granularity, `true` to generate day of week coefficients for daily data, `false` to omit day of week coefficients, which can be preferable if you are seeing too much day to day swing in the results. For weekly data, set this to null. Defaults to null.
* `seed`: (optional) fixed seed for random numbers in the MCMC process. When this isn't set, a new seed is generated each time, so model coefficients will be slightly different even for the same data. Note that JAX [uses seeds differently to numpy](https://jax.readthedocs.io/en/latest/jax-101/05-random-numbers.html).
* `custom_priors`: (optional) Bayesian priors for model parameters. You might set these if you have other data, like experimental results, that give you a prior belief for the distribution of a parameter. See [lightweight_mmm's doc on custom priors](https://lightweight-mmm.readthedocs.io/en/latest/custom_priors.html).
* `log_scale_target`: experimental feature.  Should be set to `false`.
* `raw_data_granularity`: `daily` if your CSV has one row per day; `weekly` if it has one row per week.
* `data_rows`: specify a `start_date` and `end_date` to control the time range fed into the model.
* `train_test_ratio`: (optional) ratio of data rows used for model training versus testing. Defaults to 0.9 (90% train, 10% test).
* `ignore_cols`: (optional) list of columns from your CSV that should be ignored.
* `date_col`: (optional) name of the date column in your CSV.  Dates should be in ISO 8601 format (YYYY-MM-DD).  Defaults to "`date`".
* `target_col`: name of the column with the target (output) metric.
* `extra_features_cols`: (optional) list of extra feature columns.  Extra features are factors external to marketing that influence the target.
* `media`: one block per media input metric to include in your model.  References a display name (used in charts) and column names for impressions and for spend.  Spend can be zero for non-paid marketing activities (e.g. email marketing), but when spend is zero a fixed prior should be set via `fixed_cost_prior`.  Finally, it is possible to incorporate learnings from a past model by passing a `learned_prior`.  This will be used as the scale of a `HalfNormal` distribution, so consult the appropriate distribution when setting.

Every column in the dataset should be mentioned in the config file; if you don't want to use a column in your model, add it to `ignore_cols`.

For an example, see `examples/sample_config.yaml`.