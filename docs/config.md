## Config

To run MMM, you'll need to create a `yaml` config file with the following parameters:
* `library`: at present only `lightweight_mmm` is supported.
* `model_name`: specifies the underlying mathematical formula used by the model.  One of `adstock`, `hill_adstock`, and `carryover`.
* `number_warmup`: number of samples taken during the warm up phase.  These samples will be discarded.  We recommend a value of at least 1000.
* `number_samples`: number of samples taken during the modeling phase.  These samples will be used by the model.  We recommend a value of at least 1000.
* `degrees_seasonality`: integer number of degrees of seasonality.  Larger numbers indicate more nested seasonality effect.  We recommend starting with 1, 2, or 3.
* `weekday_seasonality`: controls how the model handles day of week changes.  `null` to automatically derive the value, `true` to generate day of week coefficients for daily data, `false` to omit day of week coefficients.
* `seed`: fixed seed for random numbers in the MCMC process. When this isn't set, a new seed is generated each time, so model coefficients will be slightly different even for the same data. Note that JAX [uses seeds differently to numpy](https://jax.readthedocs.io/en/latest/jax-101/05-random-numbers.html).
* `log_scale_target`: experimental feature.  Should be set to `false`.
* `raw_data_granularity`: `daily` if your CSV has one row per day; `weekly` if it has one row per week.
* `data_rows`: specify a `start_date` and `end_date` to control the time range fed into the model.
* `ignore_cols`: (optional) list of columns from your CSV that should be ignored.
* `date_col`: (optional) name of the date column in your CSV.  Dates should be in ISO 8601 format (YYYY-MM-DD).  Defaults to "`date`".
* `target_col`: name of the column with the target (output) metric.
* `extra_features_cols`: (optional) list of extra feature columns.  Extra features are factors external to marketing that influence the target.
* `media`: one block per media input metric to include in your model.  References a display name (used in charts) and column names for impressions and for spend.  Spend can be zero for non-paid marketing activities (e.g. email marketing).  Every column in the dataset should be mentioned in the config file; if you don't want to use a column in your model, add it to `ignore_cols`.

For an example, see `examples/sample_config.yaml`.