import click
import multiprocessing
import os
import sys

os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={multiprocessing.cpu_count()}"
sys.path.append(os.path.join(os.path.dirname(__file__), "impl", "lightweight_mmm"))

from base_driver import MMMBaseDriver


@click.command()
@click.option("-c", "--config_filename", required=True, help="yaml config file")
@click.option("-f", "--input_filename", required=True, help="input data filename")
def main(config_filename: str, input_filename: str):
    driver = MMMBaseDriver()
    driver.main(config_filename, input_filename)


if __name__ == "__main__":
    main()
