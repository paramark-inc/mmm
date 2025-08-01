## Marketing Mix Models

This repo provides a wrapper around Google's lightweight_mmm (and in future, other open-source MMM products). It makes it simple to:

* Run MMM from a standard CSV file
* Start with sensible defaults, then experiment further
* Automatically produce a range of useful plots, collated into timestamped directories
<!--- * Remove outliers according to simple rules -->


## Setting up

Initialize the submodules:

```
git submodule update --init --recursive
```

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

* **Check your data**
  * See [1-validate.ipynb](./examples/jupyter/1-validate.ipynb) and [2-describe.ipynb](./examples/jupyter/2-describe.ipynb) for checks and visualizations you can run to understand and improve data hygiene.
* **Write a config file** specifying your data format and hyperparameters for the model.
  * See [sample_config.yaml](./examples/sample_config.yaml) for an example.
  * See [config.md](./docs/config.md) for documentation on config parameters.
* **Run the code.**
  * `python3 run.py` is the main entrypoint.
  * Use `-f` to pass the path to your CSV data file.
  * Use `-c` to pass the path to your config file.
* **Review results.**
  * By default, output files will be created in a directory called `results/` inside the repo.
  * Start with `model_fit_in_sample.png`, showing model fit for the training set. You don't expect this fit to be perfect, but you do expect the model to pick up trends and be able to predict the general shape of the curve.
  * Model fit for the train/holdout set is shown in `model_fit_out_of_sample.png`.
  * See [3-analyze.ipynb](./examples/jupyter/3-analyze.ipynb) for analyses you can do to improve your model fit.
  * The model's estimate of cost per target unit for each channel is shown in `media_cost_per_target_median.png` and `media_cost_per_target_mean.png`.
  * The attribution chart in `weekly_media_and_baseline_contribution.png` shows how much of the predicted target was attributed by the model to each of your marketing channels, over time.
  * See [results.md](./docs/results.md) for more information about the other files in the `results/` directory.


## Running tests

Simply run `pytest`, excluding tests in the submodules:

```
pytest test/auto/
```


## Further reading

For additional background on running MMM, these articles may be useful:
* Robyn's [analyst guide to MMM](https://facebookexperimental.github.io/Robyn/docs/analysts-guide-to-MMM/)
* Mario Filho's post [How To Create A Marketing Mix Model With LightweightMMM](https://forecastegy.com/posts/how-to-create-a-marketing-mix-model-with-lightweightmmm/)
* Lightweight MMM's [demo notebook](https://github.com/paramark-inc/lightweight_mmm/blob/main/examples/simple_end_to_end_demo.ipynb)



## Dependencies

Python dependencies are listed in the `requirements.txt` file.

This repository also includes forked copies of these other open-source projects as submodules:
* [lightweight_mmm](https://github.com/google/lightweight_mmm) (Apache License, Version 2.0)
<!---
* [orbit](https://github.com/uber/orbit) (Apache License, Version 2.0)
* [Robyn](https://github.com/facebookexperimental/Robyn) (MIT License)
-->


## License

Copyright 2023-24 Paramark, Inc.

Licensed under the Apache License, Version 2.0 (the “License”);
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an “AS IS” BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
