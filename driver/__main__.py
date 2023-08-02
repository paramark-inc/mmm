import click

from driver.driver import MMMDriver


@click.command()
@click.option("-c", "--config_filename", required=True, help="yaml config file")
@click.option("-f", "--input_filename", required=True, help="input data filename")
def main(config_filename: str, input_filename: str):
    driver = MMMDriver()
    driver.main(config_filename, input_filename)


if __name__ == "__main__":
    main()
