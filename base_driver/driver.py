from .config import git_sha, load_config

from mmm.data import DataToFit, InputData
from mmm.describe.describe import (
    describe_input_data,
    describe_config,
    describe_mmm_training,
    describe_mmm_prediction,
)
from mmm.fit.fit import fit_lightweight_mmm
from mmm.parser.csv import parse_csv_generic
from mmm.store.store import make_results_dir, load_model, save_model
from mmm.transform.transform import transform_input_generic

from impl.lightweight_mmm.lightweight_mmm.lightweight_mmm import LightweightMMM


class MMMBaseDriver:
    def load_config(self, config_filename):
        return load_config(config_filename)

    def init_output(self, data_dir: str = ".") -> str:
        return make_results_dir(data_dir=data_dir)

    def ingest_data(self, input_path: str, config: dict) -> InputData:
        data_dict = parse_csv_generic(input_path, config)
        input_data = transform_input_generic(data_dict, config)
        return input_data

    def run_feature_engineering(self, input_data: InputData, _config: dict) -> InputData:
        # No-op in this version. This method provides a place for data transformations
        # such as derived seasonality, resampling by date, or complex scaling.
        return input_data

    def describe_data(
        self,
        results_dir: str,
        input_data_raw: InputData,
        input_data_processed: InputData,
        raw_config: bytes,
        current_commit: str,
    ) -> None:
        input_data_raw.dump(output_dir=results_dir, suffix="raw", verbose=True)
        describe_input_data(input_data=input_data_raw, results_dir=results_dir, suffix="raw")
        describe_config(output_dir=results_dir, config=raw_config, git_sha=current_commit)

        input_data_processed.dump(output_dir=results_dir, suffix="processed", verbose=True)
        describe_input_data(
            input_data=input_data_processed, results_dir=results_dir, suffix="processed"
        )

    def fit(
        self,
        results_dir: str,
        input_data_processed: InputData,
        config: dict,
        model_filename: str = None,
    ) -> (DataToFit, LightweightMMM):
        data_to_fit = DataToFit.from_input_data(input_data=input_data_processed, config=config)

        if model_filename:
            model = load_model(model_filename=model_filename)
        else:
            model = fit_lightweight_mmm(
                config=config,
                data_to_fit=data_to_fit,
                results_dir=results_dir,
            )

        return data_to_fit, model

    def visualize(
        self,
        results_dir: str,
        model: LightweightMMM,
        input_data_processed: InputData,
        data_to_fit: DataToFit,
        config: dict,
    ) -> None:
        describe_mmm_training(
            mmm=model,
            input_data=input_data_processed,
            data_to_fit=data_to_fit,
            degrees_seasonality=config["degrees_seasonality"],
            results_dir=results_dir,
            include_response_curves=False,
        )
        describe_mmm_prediction(mmm=model, data_to_fit=data_to_fit, results_dir=results_dir)

    def save_model(self, results_dir: str, model: LightweightMMM, data_to_fit: DataToFit) -> None:
        data_to_fit.dump(results_dir=results_dir)
        save_model(mmm=model, results_dir=results_dir)

    def main(self, config_filename, input_filename):
        config_raw, config = self.load_config(config_filename)
        current_commit = git_sha()

        results_dir = self.init_output()
        input_data_raw = self.ingest_data(input_filename, config)
        input_data_processed = self.run_feature_engineering(input_data_raw, config)
        self.describe_data(
            results_dir, input_data_raw, input_data_processed, config_raw, current_commit
        )
        data_to_fit, model = self.fit(results_dir, input_data_processed, config)
        self.visualize(results_dir, model, input_data_processed, data_to_fit, config)
        self.save_model(results_dir, model, data_to_fit)
